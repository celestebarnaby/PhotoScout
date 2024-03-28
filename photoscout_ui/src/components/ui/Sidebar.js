import React, { useState } from 'react';
import Box from '@mui/material/Box';

import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ImageList from '@mui/material/ImageList';
import ImageListItem from '@mui/material/ImageListItem';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import Divider from '@mui/material/Divider';


function Sidebar({ allFiles, changeImage, savedImages, handleTextChange, handleTextSubmit, exampleImages, tags }) {

    return (
        <Box sx={{ height: "100%" }} className="sidebar">
            <Button sx={{ marginTop: 1, marginBottom: 1 }} fullWidth variant="contained" onClick={handleTextSubmit}>Search </Button>
            <TextField
                fullWidth
                id="outlined-name"
                label="Describe images to search"
                variant="outlined"
                sx={{ background: "white" }}
                onChange={handleTextChange}
                autoComplete='off'
            />
            <h3>Example Images</h3>
            {Object.keys(exampleImages).length > 0 ?
                <Box sx={{ paddingRight: "30px", height: "auto" }}>
                    <ImageList sx={{ margin: "8px", width: "100%", height: "calc(100% - 76px)" }} cols={3} rowHeight={164}>
                        {Object.keys(exampleImages).map(img => {
                            let class_name = exampleImages[img] ? "example-img-pos" : "example-img-neg";
                            return <ImageListItem key={img} onClick={() => changeImage(img)}>
                                <img className={class_name}
                                    src={`${img.replace("photoscout_ui/public/", "./")}`}
                                    loading="lazy"
                                />
                            </ImageListItem>
                        })}
                    </ImageList>
                </Box> : <div>Add example images to refine search.</div>
            }
            <Divider></Divider>
            {/* <Divider /> */}
            {AllImages(allFiles, savedImages, changeImage)}
        </Box>
    );
}

function AllImages(allFiles, savedImages, changeImage) {
    // let height = imgsToAnnotate.length > 0 ? 200 : 0;

    return <ImageList sx={{ width: "100%", height: "87%" }} cols={3} rowHeight={164}>
        {allFiles.map(img => {
            let class_name = savedImages.includes(img) ? "grayed-out" : "";
            return <ImageListItem key={img} onClick={() => changeImage(img)}>
                <img
                    src={`${img.replace("photoscout_ui/public/", "./")}`}
                    className={class_name}
                    loading="lazy"
                />
            </ImageListItem>
        })}
    </ImageList>
}

export default Sidebar;