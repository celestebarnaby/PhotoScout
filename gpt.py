import openai
from interpreter import *
from utils import *
from dsl import *
from synthesizer import *
import regex as re

with open("../../openai-key.txt") as f:
    sk = f.read().strip()

openai.api_key = sk


class Hole:
    def __init__(self, node_type, val=None):
        self.node_type = node_type
        self.val = val

    def __str__(self):
        return type(self).__name__

    def duplicate(self):
        return Hole(self.node_type, self.val)

    def __lt__(self, other):
        if not isinstance(other, Hole):
            return False
        return str(self) < str(other)


class Tree:
    def __init__(self):
        # self.id: int = _id
        self.nodes: Dict[int, Formula] = {}
        self.to_children: Dict[int, List[int]] = {}
        self.to_parent: Dict[int, int] = {}
        # self.node_id_counter = itertools.count(0)
        # self.depth = 1
        # self.size = 1
        self.var_nodes = []
        self.holes_to_vals = {}

    def duplicate(self) -> "Tree":
        ret = Tree()
        # ret.nodes = copy.copy(self.nodes)
        ret.nodes = {}
        for key, val in self.nodes.items():
            if isinstance(val, Hole) or isinstance(val, Node):
                ret.nodes[key] = val.duplicate()
            else:
                ret.nodes[key] = val
        ret.to_children = self.to_children.copy()
        ret.to_parent = self.to_parent.copy()
        # ret.node_id_counter = itertools.tee(self.node_id_counter)[1]
        ret.var_nodes = self.var_nodes.copy()
        ret.holes_to_vals = self.holes_to_vals.copy()
        # ret.depth = self.depth
        # ret.size = self.size
        return ret

    def __lt__(self, other):
        if self.size == other.size and self.depth == other.depth:
            return self.id < other.id
        if self.size == other.size:
            return self.depth < other.depth
        return self.size < other.size


class Benchmark:
    def __init__(self, gt_prog, desc, dataset_name, example_imgs=[]):
        self.gt_prog = gt_prog
        self.desc = desc
        self.dataset_name = dataset_name
        self.example_imgs = example_imgs


def clean_img(img):
    for obj in img.values():
        for key in {
            "Prevmost",
            "Nextmost",
            "Leftmost",
            "Rightmost",
            "Topmost",
            "Bottommost",
            "Hash",
            "Emotions",
            "ObjPosInImgLeftToRight",
            "ImgIndex",
        }:
            if key in obj:
                del obj[key]
        obj["Bounding Box"] = {
            "Left": obj["Loc"][0],
            "Right": obj["Loc"][2],
            "Top": obj["Loc"][1],
            "Bottom": obj["Loc"][3],
        }
        del obj["Loc"]
        if "AgeRange" in obj:
            obj["Age"] = obj["AgeRange"]["High"]
            del obj["AgeRange"]
    return list(img.values())


def camel_case_to_words(s):
    # Use regular expressions to split the input string at camelCase boundaries
    # and then join the words with spaces and convert to lowercase.
    words = re.findall(r"[A-Z][a-z]*|[a-z]+", s)
    transformed_string = " ".join(words).lower()
    return transformed_string


def insert_and_after_last_comma(string):
    parts = string.rsplit(",", 1)
    if len(parts) > 1:
        return f"{parts[0]}, and {parts[1].strip()}"
    else:
        return string


def ask_for_hole_expl(holes):
    hole_str = ", ".join(holes)
    hole_str = insert_and_after_last_comma(hole_str)
    plural = "s" if len(holes) > 1 else ""
    expl = """
I don't know the term{} {}. Can you add a few examples and/or tags to show me what you mean?
""".format(
        plural, hole_str
    )
    return expl


def get_hole_expl_for_top_prog(holes_to_vals):
    holes = ['"{}"'.format(hole) for hole in holes_to_vals.keys()]
    hole_str = ", ".join(holes)
    hole_str = insert_and_after_last_comma(hole_str)
    plural = "s" if len(holes) > 1 else ""
    holes_and_vals = ['"{}" is "{}"'.format(k, v) for (k, v) in holes_to_vals.items()]
    holes_and_vals_str = ", ".join(holes_and_vals)
    holes_and_vals_str = insert_and_after_last_comma(holes_and_vals_str)
    expl = """

I don't know the term{} {} in your query. Based on your example images, I assume that {}. If I have made a mistake, please add some more example images to clarify what you mean.
""".format(
        plural, hole_str, holes_and_vals_str
    )
    return expl


def prog_to_expl(prog):
    if isinstance(prog, ForAll):
        return "for each object {} that I can see in the image, {}".format(
            prog.var, prog_to_expl(prog.subformula)
        )
    if isinstance(prog, Exists):
        return "I can see an object {} in the image such that {}".format(
            prog.var, prog_to_expl(prog.subformula)
        )
    if isinstance(prog, And):
        return "{} and {}".format(
            prog_to_expl(prog.subformula1), prog_to_expl(prog.subformula2)
        )
    if isinstance(prog, IfThen):
        return "if {}, then {}".format(
            prog_to_expl(prog.subformula1), prog_to_expl(prog.subformula2)
        )
    if isinstance(prog, Not):
        return "it is not the case that {}".format(prog_to_expl(prog.subformula))
    if isinstance(prog, Is):
        return "{} is a {}".format(prog.var1, prog.var2)
    if isinstance(prog, IsAbove):
        return "{} is above {}".format(prog.var1, prog.var2)
    if isinstance(prog, IsLeft):
        return "{} is left of {}".format(prog.var1, prog.var2)
    if isinstance(prog, IsNextTo):
        return "{} is next to {}".format(prog.var1, prog.var2)
    if isinstance(prog, IsNextTo):
        return "{} is inside {}".format(prog.var1, prog.var2)


def make_text_query(query, env, examples, tags, cache, use_cache):
    objects = get_objects(env)
    objects = objects.union(tags)
    output_trees = []
    print(objects)
    if use_cache and query in cache["progs"]:
        texts = cache["progs"][query]
    else:
        example_progs = [
            (
                "Every person is next to a cat",
                "ForAll x.Exists y.Is(y, cat) And Is(x, person) -> IsNextTo(x, y)",
            ),
            (
                "The image contains a face that is smiling, and has their eyes open",
                "Exists x.Is(x, smilingFace) And Is(x, eyesOpenFace)",
            ),
            (
                "Alice is in the image and everyone is smiling",
                "Exists x.ForAll y.Is(x, Alice) And Is(y, face) -> Is(y, smilingFace)",
            ),
            (
                "The image contains a cat inside a box.",
                "Exists x.Exists y.Is(y, cat) And Is(x, box) And IsInside(y, x)",
            ),
            ("There is a tree in the image.", "Exists x.Is(x, Tree)"),
            (
                "The image contains a chair and a table",
                "Exists x.Exists y.Is(x, chair) And Is(y, table)",
            ),
            (
                "The image contains a chair to the left of a table",
                "Exists x.Exists y.Is(x, chair) And Is(y, table) And IsLeft(x, y)",
            ),
            (
                "All faces do not have eyes open",
                "ForAll x.Is(x, face) -> Not Is(x, eyesOpenFace)",
            ),
        ]
        message_text = ""
        for prog in example_progs:
            message_text += "task: {}\nprogram:{}\n\n".format(prog[0], prog[1])
        query_content = "task: {}\nprogram: ".format(query)
        message = {"role": "system", "content": message_text + query_content}
        try:
            gpt_output = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", temperature=0.9, messages=[message], n=20
            )
        except:
            return ([], "There was a server error. Try again!", "", None)
        output_trees = []
        texts = [
            choice["message"]["content"].strip() for choice in gpt_output["choices"]
        ]
        cache[query] = texts
    for text in texts:
        tree = Tree()
        # We only take the progs that parse
        res, holes = parse_formula(text, tree, objects, [])
        if res:
            output_trees.append((tree, text, holes))

    print("GPT output:")
    print(texts)
    print("Num outputs parsed: {}".format(str(len(output_trees))))
    print()

    if not output_trees:
        return ([], "I didn't understand your query.", "", None)

    top_prog = None
    prog_without_holes = False
    example_imgs = [tup[0] for tup in examples]
    env_objects = get_objects(env, example_imgs)
    holes_to_vals = None
    for tree, _, holes in output_trees:
        # prog = construct_prog_from_tree(tree)
        # print(prog)
        if len(tree.var_nodes) > 0:
            prog, holes_to_vals = fill_in_holes(tree, examples, env, env_objects)
            if prog is not None:
                top_prog = prog
                break
        else:
            prog = construct_prog_from_tree(tree)
            prog_without_holes = True
            matching = True
            for img, output in examples:
                if eval_prog(prog, env[img]["environment"]) != output:
                    matching = False
                    break
            if matching:
                top_prog = prog
                break

    if not top_prog:
        if not examples:
            if prog_without_holes:
                return (
                    [],
                    "I am confused by your query. Could you try writing it in a different way, or adding some example images?",
                    "",
                    None,
                )
            holes = output_trees[0][2]
            return [], ask_for_hole_expl(set(holes)), "", None
        return (
            [],
            "I don't think your query matches your example images. Can you replace some of your examples with different images, or edit your text query?",
            "",
            None,
        )

    matching_imgs = []
    print("Top prog:")
    print(top_prog)
    print()
    for img, img_env in env.items():
        if eval_prog(top_prog, img_env["environment"]):
            matching_imgs.append(img)
    print(len(matching_imgs))

    if use_cache and str(top_prog) in cache["expls"]:
        expl = cache["expls"][str(top_prog)]
    else:
        expl_query = """
        input: For each object x that I can see in the image, I can see an object y in the image such that if y is cat and x is person, then x is next to y.
        output: I found all images where every person I can see is next to a cat.

        input: I can see an object x in the image such that x is smilingFace and x is eyesOpenFace.
        output: I found all images where I can see a face that is smiling, and has their eyes open.

        input: I can see an object x in the image such that for each object y that I can see in the image, x is id23 and if y is face, then y is smiling face.
        output: I found all images where I see face #23 and everyone I can see is smiling.

        input: I can see an object x in the image such that I can see an object y in the image such that x is id57 and y is id6.
        output: I found all images where I see face #57 and face #5.

        input: I can see an object x in the image such that I can see an object y in the image such that y is cat and x is box and y is inside x.  
        output: I found all images where I can see a cat inside a box.

        input: I can see an object x in the image such that I can see an object y in the image such that x is chair and y is table.
        output: I found all images where I can see a chair and a table.

        input: For each object x that I can see in the image, if x is face then it is not that case that x is smilingFace.
        output:  I found all images where no faces that I can see are smiling.

        input: I can see an object x in the image such that x is a cat and x is a box
        output: I found all images where there is a cat that is a box.

        input: I can see an object x in the image such that if x is person then x is person
        output: I found all images where there is a person who is a person.

        input: {}
        output:
    """.format(
            prog_to_expl(top_prog)
        )

        expl_message = {"role": "system", "content": expl_query}
        try:
            expl_output = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=[expl_message]
            )
            expl = (expl_output["choices"][0]["message"]["content"],)
        except:
            return [], "There was a server error. Try again!", "", None
    if holes_to_vals:
        hole_expl = get_hole_expl_for_top_prog(holes_to_vals)
    else:
        hole_expl = ""

    return (
        matching_imgs,
        expl,
        hole_expl,
        str(top_prog),
    )


str_to_pred = {
    "Is": Is,
    "IsAbove": IsAbove,
    "IsLeft": IsLeft,
    "IsNextTo": IsNextTo,
    "IsInside": IsInside,
}


def parse_formula(formula, tree, objects, holes, parent_node_num=None, used_vars=[]):
    formulas = [
        ("^ForAll (\w*).(.*)$", ForAll),
        ("^Exists (\w*).(.*)$", Exists),
    ]
    one_param_subformulas = [
        ("^Not (.*)$", Not),
    ]
    two_param_subformulas = [
        ("(?<=^(.*)) -> (?=(.*)$)", IfThen),
        ("(?<=^(.*)) And (?=(.*)$)", And),
    ]
    predicate = "^(Is\w*)\((\w*), (\w*)\)$"
    new_node_num = len(tree.nodes)
    for regex, f in formulas:
        m = re.search(regex, formula)
        if m is not None:
            var = m.group(1)
            used_vars.append(var)
            subformula = m.group(2)
            new_node = f(var, None)
            tree.nodes[new_node_num] = new_node
            tree.to_children[new_node_num] = []
            successful_parse, _ = parse_formula(
                subformula, tree, objects, holes, new_node_num, used_vars
            )
            if successful_parse:
                if parent_node_num is not None:
                    tree.to_parent[new_node_num] = parent_node_num
                    tree.to_children[parent_node_num].append(new_node_num)
                return True, holes
    for regex, f in one_param_subformulas:
        m = re.search(regex, formula)
        if m is not None:
            subformula = m.group(1)
            new_node = f(None)
            tree.nodes[new_node_num] = new_node
            tree.to_children[new_node_num] = []
            successful_parse, _ = parse_formula(
                subformula, tree, objects, holes, new_node_num, used_vars
            )
            if successful_parse:
                if parent_node_num is not None:
                    tree.to_parent[new_node_num] = parent_node_num
                    tree.to_children[parent_node_num].append(new_node_num)
                return True, holes
    for regex, f in two_param_subformulas:
        m = re.findall(regex, formula)
        for subformula1, subformula2 in m:
            new_node = f(None, None)
            tree.nodes[new_node_num] = new_node
            tree.to_children[new_node_num] = []
            (successful_parse1, _), (successful_parse2, _) = parse_formula(
                subformula1, tree, objects, holes, new_node_num, used_vars
            ), parse_formula(subformula2, tree, objects, holes, new_node_num, used_vars)
            if successful_parse1 and successful_parse2:
                if parent_node_num is not None:
                    tree.to_parent[new_node_num] = parent_node_num
                    tree.to_children[parent_node_num].append(new_node_num)
                return True, holes
    m = re.search(predicate, formula)
    if m is not None:
        pred_str = m.group(1)
        var1, var2 = m.group(2), m.group(3)
        if pred_str in str_to_pred:
            f = str_to_pred[pred_str]
            new_node = f(var1, var2)
        else:
            holes.append('"' + camel_case_to_words(pred_str) + '"')
            new_node = Hole("relation")
            tree.var_nodes.append(new_node_num)
        # new_node = f(var1, var2)
        tree.nodes[new_node_num] = new_node
        tree.to_children[new_node_num] = []
        for var in [var1, var2]:
            new_child_node_num = len(tree.nodes)
            if (
                var in used_vars
                or var.lower() in objects
                or var in {"smilingFace", "eyesOpenFace"}
            ):
                new_child_node = var
            else:
                new_child_node = Hole("var", var)
                holes.append('"' + camel_case_to_words(var) + '"')
                tree.var_nodes.append(new_child_node_num)
            tree.nodes[new_child_node_num] = new_child_node
            tree.to_children[new_node_num].append(new_child_node_num)
            tree.to_parent[new_child_node_num] = new_node_num
        if parent_node_num is not None:
            tree.to_parent[new_node_num] = parent_node_num
            tree.to_children[parent_node_num].append(new_node_num)
        return True, holes
    return False, holes


def test_parser():
    tests = [
        (
            "Exists x.Exists y.Is(y, cat) And Is(x, box) And IsInside(y, x)",
            Exists(
                "x",
                Exists(
                    "y", And(Is("y", "cat"), And(Is("x", "box"), IsInside("y", "x")))
                ),
            ),
            ["cat", "box"],
        ),
        (
            "ForAll x.Is(x, person) -> IsAbove(x, bicycle)",
            ForAll("x", IfThen(Is("x", "person"), IsAbove("x", "bicycle"))),
            ["person", "bicycle"],
        ),
        (
            "ForAll x.Is(x, person) -> IsAbove(x, bike)",
            ForAll("x", IfThen(Is("x", "person"), IsAbove("x", Hole("var")))),
            ["person", "bicycle"],
        ),
        (
            "Exists x.Is(x, smilingFace) And Is(x, eyesOpenFace)",
            Exists("x", And(Is("x", "smilingFace"), Is("x", "eyesOpenFace"))),
            ["smilingFace", "eyesOpenFace"],
        ),
        (
            "Exists x.ForAll y.Is(x, Alice) And Is(y, face) -> Is(y, smilingFace)",
            Exists(
                "x",
                ForAll(
                    "y",
                    And(
                        Is("x", "Alice"),
                        IfThen(Is("y", "face"), Is("y", "smilingFace")),
                    ),
                ),
            ),
            ["alice", "face", "smilingFace"],
        ),
        (
            "Exists x.Is(x, face) And Not Is(x, smilingFace)",
            Exists("x", And(Is("x", "face"), Not(Is("x", "smilingFace")))),
            ["face", "smilingFace"],
        ),
        (
            "ForAll x.Is(x, bicycle) -> IsAbove(person, x)",
            ForAll("x", IfThen(Is("x", "bicycle"), IsAbove("person", "x"))),
            ["bicycle", "person"],
        ),
        (
            "ForAll x.Is(x, bicycle) -> IsRiding(person, x)",
            ForAll("x", IfThen(Is("x", "bicycle"), Hole("relation"))),
            ["bicycle", "person"],
        ),
        (
            "Exists x.Is(x, car) And IsInside(x, person)",
            Exists("x", And(Is("x", "car"), IsInside("x", "person"))),
            ["car", "person"],
        ),
        (
            "Exists x.Is(x, Alice) And IsLeft(Bob, x)",
            Exists("x", And(Is("x", Hole("var")), IsLeft(Hole("var"), "x"))),
            [],
        ),
        ("Exists x.Is(x, Alice)", Exists("x", Is("x", "Alice")), ["alice"]),
        (
            "Exists x.Exists y.Is(x, Alice) And Is(y, Bob)",
            Exists("x", Exists("y", And(Is("x", "Alice"), Is("y", "Bob")))),
            ["alice", "bob"],
        ),
        (
            "Exists x.Is(x, Alice) And IsNextTo(x, Bob)",
            Exists("x", And(Is("x", "Alice"), IsNextTo("x", "Bob"))),
            ["alice", "bob"],
        ),
        (
            "ForAll x.Is(x, face) -> Not Is(x, smilingFace)",
            ForAll("x", IfThen(Is("x", "face"), Not(Is("x", "smilingFace")))),
            ["face", "smilingFace"],
        ),
    ]
    for test in tests:
        tree = Tree()
        parse_formula(test[0], tree, test[2], [])
        prog = construct_prog_from_tree(tree)
        if str(prog) != str(test[1]):
            print("FAIL")
            print(test[0])
            print(str(prog))
            print(str(test[1]))
            print()


def test_synthesizer():
    tests = [
        (
            "Exists x.(Exists y.((Is(x, car)) And ((Is(y, person)) And (IsInside(y, x)))))",
            "objects",
            [
                (
                    "image-eye-web/public/images/objects/5421932595_68f7ab545d_c.jpg",
                    True,
                ),
                (
                    "image-eye-web/public/images/objects/3000363792_ded885dd2f_c.jpg",
                    False,
                ),
            ],
        ),
        (
            "Exists x.(Exists y.((Is(x, person)) And ((Is(y, bicycle)) And (IsRiding(x, y)))))",
            "objects",
            [
                ("image-eye-web/public/images/objects/9107010_479657625c_o.jpg", True),
                (
                    "image-eye-web/public/images/objects/3000363792_ded885dd2f_c.jpg",
                    False,
                ),
            ],
        ),
    ]
    for test in tests:
        tree = Tree()
        res = parse_formula(test[0], tree, ["person", "bicycle"], [])
        if not res:
            print("PARSING FAILED")
            raise TypeError
        img_folder = "image-eye-web/public/images/" + test[1] + "/"
        img_to_env, _ = preprocess(img_folder)
        examples = test[2]
        prog = fill_in_holes(tree, examples, img_to_env, get_objects(img_to_env))
        print(prog)


if __name__ == "__main__":
    # test_parser()
    test_synthesizer()
