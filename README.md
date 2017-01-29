# Characters recognition

Little program to classify english characters, handwriting in a window. It is implemented in C++, using the OpenCV library.

Run the program and write in the opened window, then press <kbd>R</kbd> and watch in the command prompt the output. Press <kbd>E</kbd> to reset the canvas and write again.


### Train model

The model is already trained and it is stored in the file `svm10.yml`. 

But if you want to train again, pass the argument `-train` when running the program.
The dataset can be downloaded from [chars74k website](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/).
But the structure should be the following:

```
data
|__ Fnt
    | Sample011
        | img011-00001.png
        | img011-00002.png
        | ...
    | Sample012
        | img012-00001.png
        | img012-00002.png
        | ...
    | ...
|__ Hnd
    | Sample011
        | img011-001.png
        | img011-002.png
        | ...
    | Sample012
        | img012-001.png
        | img012-002.png
        | ...
    | ...
|__ Img
    | Sample011
        | img011-00001.png
        | img011-00002.png
        | ...
    | Sample012
        | img012-00001.png
        | img012-00002.png
        | ...
    | ...
```

The names of the images don't matter, but they must be `png`.