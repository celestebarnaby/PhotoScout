# PhotoScout: Synthesis-Powered Multi-Modal Image Search

This is the codebase for the paper ["PhotoScout: Synthesis-Powered Multi-Modal Image Search."](https://arxiv.org/abs/2401.10464)

## How to Run PhotoScout

1. [Install Node.js and npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)
2. Run `pip install -r requirements.txt`.
3. Put your [OpenAI Secret API Key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key) in a file called `openai-key.txt` one directory up from the `PhotoScout` directory
4. From the `photoscout_ui` directory, run `npm install` and then `npm start`.
5. In a separate terminal (in the main `PhotoScout` directory), run `python app.py`. 
6. You can also run `python app.py --use_cache` to see the results referenced in the usage example in the paper. In particular, the GPT output for the text query `"Alice next to Bob holding flowers"` are cached.
7. The interface should open in your browser.

You can use PhotoScout as described in the paper, using the datasets included in the user study. The practice task uses the transportation dataset, tasks 1 and 2 use the music festival dataset, and tasks 3 and 4 use the wedding dataset.

## How to Run CLIPWrapper

1. [Install Node.js and npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)
2. Run `pip install -r requirements.txt`.
3. From the `baseline_ui` directory, run `npm install` and then `npm start`.
4. In a separate terminal (in the main `PhotoScout` directory), run `python baseline_app.py`. 
5. The interface should open in your browser.

You can use CLIPWrapper as described in the paper, using the datasets included in the user study. The practice task uses the transportation dataset, tasks 1 and 2 use the music festival dataset, and tasks 3 and 4 use the wedding dataset.
