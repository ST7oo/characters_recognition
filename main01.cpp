

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/ml.hpp>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <sys/types.h>
#include <dirent.h>
#include <numeric>

using namespace cv;
using namespace std;

// Global variables
Mat imagen, screenBuffer;
int radius, last_x, last_y, new_size, n_classes;
bool drawing, fnt, hnd, img;
const int NUMBERS = 0;
const int UPPERS = 10;
const int LOWERS = 36;
int from_class = UPPERS;


// return a string with zeros before the number
string fill_zeros(int num, int num_zeros)
{
	stringstream ss;
	ss << num;
	string ret;
	ss >> ret;
	int str_length = ret.length();
	for (int i = 0; i < num_zeros - str_length; i++)
		ret = "0" + ret;
	return ret;
}

// preprocess image
void preprocess(Mat image, Mat &row_data, int c=0) {
    Mat binary, resized, result, vect;
    int black_dots = 0;

    threshold(image, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

    resize(binary, resized, Size(new_size,new_size), 0, 0, INTER_NEAREST);

    for(int i=0; i<new_size; i++){
        if(resized.at<uchar>(0,i) == 0){
            black_dots++;
        }
        if(resized.at<uchar>(1,i) == 0){
            black_dots++;
        }
    }
    for(int i=2; i<new_size; i++){
        if(resized.at<uchar>(i,0) == 0){
            black_dots++;
        }
        if(resized.at<uchar>(i,1) == 0){
            black_dots++;
        }
        if(resized.at<uchar>(new_size-1,i) == 0){
            black_dots++;
        }
        if(resized.at<uchar>(new_size-2,i) == 0){
            black_dots++;
        }
    }
    for(int i=2; i<new_size-2; i++){
        if(resized.at<uchar>(i,new_size-1) == 0){
            black_dots++;
        }
        if(resized.at<uchar>(i,new_size-2) == 0){
            black_dots++;
        }
    }
    if (black_dots > ((new_size*8)-12)/2.5){
        threshold(resized, result, 140, 255, THRESH_BINARY_INV);
    }
    else {
        result = resized;
    }

    result.convertTo(vect, CV_32F);
    row_data = vect.reshape(1,1);
    divide(row_data, 255, row_data);
}

// read a folder
int read_folder(string folder, vector<Mat> &data) {
    string name;
    Mat image, row_data;
    DIR *dir;
    struct dirent *ent;
    int c = 0;
    if ((dir = opendir(folder.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            name = ent->d_name;
            if (name.find(".png")!=string::npos){
                image = imread(folder+name, IMREAD_GRAYSCALE);
                if(image.data) {
                    preprocess(image, row_data, c);
                    data.push_back(row_data);
                    c++;
                }
            }
        }
        closedir(dir);
    } else {
        cout<<"Unable to open directory "<<folder<<endl;
    }
    return c;
}

// function to call read folder for each type of images
void load_and_preprocess(string path, int n_folder, vector<Mat> &data, int &num_samples){
    string folder;
    int c = 0;

    if(fnt){
        folder = path+"Fnt/Sample"+fill_zeros(n_folder+1, 3)+"/";
        c += read_folder(folder, data);
    }
    if(hnd){
        folder = path+"Hnd/Sample"+fill_zeros(n_folder+1, 3)+"/";
        c += read_folder(folder, data);
    }
    if(img){
        folder = path+"Img/Sample"+fill_zeros(n_folder+1, 3)+"/";
        c += read_folder(folder, data);
    }

    num_samples = c;
}

// read data and stores in trainX, trainY, testX, testY
void read_data(string path, Mat &trainX, Mat &testX, Mat &trainY, Mat &testY, float split, int n_threads = 8){
    int train_samples, test_samples;
    vector<thread> threads(n_threads);
    vector<vector<Mat>> data(n_classes, vector<Mat>());
    vector<int> num_samples(n_classes);

    cout<<"Loading images and preprocessing..."<<endl;
    int i = 0, c = 0;
    while (i<n_classes) {
        c = 0;
        for(int j=0; j<n_threads && i<n_classes; j++) {
            threads[j] = thread(load_and_preprocess, path, i+from_class, ref(data[i]), ref(num_samples[i]));
            i++;
            c++;
        }
        for(int j=0; j<c; j++) {
            threads[j].join();
        }
    }

    cout<<"Preparing data..."<<endl;
    Mat trainData;
	Mat trainClasses;
    Mat testData;
	Mat testClasses;
	cout<<"Training samples: ";
    for(int i = 0; i < n_classes; i++) {
        train_samples = (int)round(num_samples[i]*split);
        cout<<train_samples<<", ";
        for(int j=0; j<num_samples[i]; j++){
            if(j < train_samples){
                trainClasses.push_back((float)(i));
                trainData.push_back(data[i][j]);
            }
            else {
                testClasses.push_back((float)(i));
                testData.push_back(data[i][j]);
            }
        }
    }
    cout<<"= "<<trainData.rows<<endl;
    cout<<"Testing samples: ";
    for(int i = 0; i < n_classes; i++) {
        cout<<num_samples[i] - (int)round(num_samples[i]*split)<<", ";
    }
    cout<<"= "<<testData.rows<<endl;
    trainX = trainData;
    trainY = trainClasses;
    testX = testData;
    testY = testClasses;
}

// train the SVM model
Ptr<ml::SVM> trainSVM(Mat X, Mat Y, bool save=false) {
    Y.convertTo(Y,CV_32S);
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setC(0.1);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1e3, 1e-6));
    Ptr<ml::TrainData> td = ml::TrainData::create(X, ml::ROW_SAMPLE, Y);
    svm->train(td);

    if (save){
        svm->save("svm"+to_string(n_classes)+".yml");
    }
    return svm;
}

// evaluate the prediction
float evaluate(cv::Mat& predicted, cv::Mat& actual) {
    assert(predicted.rows == actual.rows);
    int t = 0;
    int f = 0;
    for(int i = 0; i < actual.rows; i++) {
        int p = predicted.at<float>(i,0);
        int a = actual.at<float>(i,0);
        if(p == a) {
            t++;
        } else {
            f++;
        }
    }
    return (t * 1.0) / (t + f);
}

// get the label string
string get_label(int n) {
    char letters[] = "abcdefghijklmnopqrstuvwxyz";
    if (from_class == NUMBERS) {
        return to_string(n);
    }
    else if (from_class == UPPERS) {
        return string(1, toupper(letters[n]));
    }
    else {
        return string(1, letters[n]);
    }
}

// predict with the model
void predictSVM(Ptr<ml::SVM> model, Mat X, Mat Y) {
    Mat predicted(Y.rows, 1, CV_32F);
    model->predict(X, predicted);
    cout << "Accuracy = " << evaluate(predicted, Y) << endl;
}

// calculate the elapsed time
string elapsed_time(std::chrono::steady_clock::time_point start_time, std::chrono::steady_clock::time_point end_time) {
    float elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) / 1000000.0;
    if (elapsed > 60) {
        elapsed /= 60;
        return to_string(elapsed) + " min";
    } else {
        return to_string(elapsed) + " sec";
    }
}

// classify one sample
void classify(Ptr<ml::SVM> model, Mat sample) {
    Mat res, row_data;
    preprocess(sample, row_data, false);
    model->predict(row_data, res);
    cout<<get_label(res.at<float>(0))<<endl;
}

// train the model
Ptr<ml::SVM> get_model(string path_data, float split, int n_threads=1) {

    Ptr<ml::SVM> model;

    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    Mat trainX, testX, trainY, testY;

    // Reading
    cout << "\nStart reading:\n";

    start_time = std::chrono::steady_clock::now();
    read_data(path_data, trainX, testX, trainY, testY, split, n_threads);
    cout<<trainX.size()<<endl;
    cout<<testX.size()<<endl;
    end_time = std::chrono::steady_clock::now();

    cout << "Reading completed in " << elapsed_time(start_time, end_time) << endl;


    // Training
    cout << "\nStart training:\n";
    start_time = std::chrono::steady_clock::now();
    model = trainSVM(trainX, trainY);
    end_time = std::chrono::steady_clock::now();
    cout << "Training completed in " << elapsed_time(start_time, end_time) << endl;


    // Predicting
    cout << "\nStart predicting:\n";
    start_time = std::chrono::steady_clock::now();
    predictSVM(model, testX, testY);
    end_time = std::chrono::steady_clock::now();
    cout << "Predicting completed in " << elapsed_time(start_time, end_time) << endl;


    // Training and saving final model
    cout << "\nStart training and saving final model:\n";
    start_time = std::chrono::steady_clock::now();
    Mat completeX, completeY;
    vconcat(trainX, testX, completeX);
    vconcat(trainY, testY, completeY);
    model = trainSVM(completeX, completeY, true);
    end_time = std::chrono::steady_clock::now();
    cout << "Training and saving final model completed in " << elapsed_time(start_time, end_time) << endl;

    return model;
}

// load the model
Ptr<ml::SVM> get_model() {
    Ptr<ml::SVM> model = ml::StatModel::load<ml::SVM>("svm"+to_string(n_classes)+".yml");
    return model;
}

// draw a circle where the cursor is
void draw_cursor(int x, int y){
	screenBuffer=imagen.clone();
	circle(screenBuffer, Point(x,y), radius, CV_RGB(0,0,0), -1, CV_AA);
}

// draw a circle in the drawing window
void draw(int x,int y){
	circle(imagen, Point(x,y), radius, CV_RGB(0,0,0), -1, CV_AA);
	screenBuffer = imagen.clone();
	imshow("Drawing Window", screenBuffer);
}

// on mouse events
void on_mouse(int event, int x, int y, int flags, void* param) {
	last_x=x;
	last_y=y;
	draw_cursor(x,y);
	if(event==CV_EVENT_LBUTTONDOWN){
        drawing=true;
        draw(x,y);
    }
    else if(event==CV_EVENT_LBUTTONUP){
        drawing=false;
    }
	else if(event == CV_EVENT_MOUSEMOVE  &&  flags & CV_EVENT_FLAG_LBUTTON){
        if(drawing){
            draw(x,y);
        }
    }
}

// help
void help(){
    cout<<"Use the mouse to draw a character"<<endl;
    cout<<"Press 'r' to classify"<<endl;
    cout<<"Press 'e' to reset"<<endl;
    cout<<"Press '+' to increase the radius"<<endl;
    cout<<"Press '-' to decrease the radius"<<endl;
    cout<<"Press 't' to save the current image"<<endl;
    cout<<"Press 'ESC' to quit"<<endl;
    cout<<"-----------------------------------"<<endl<<endl;
}

// main
int main()
{
    // configurable variables
    string path_data = "data/"; // relative path to images
    new_size = 64; // size to resize
    n_classes = 10; // number of classes
    float split = 0.8; // percentage to split in training and testing
    int n_threads = 8; // number of cores to use
    fnt = true; // use font type images
    hnd = false; // use hand type images
    img = false; // use image type images

    // private variables
	int key;
	bool execute = true;
    drawing = false;
	radius = 7;
	last_x=last_y=0;
	imagen= Mat(Size(128,128),CV_8U, 255);
	screenBuffer=imagen.clone();

    cout << "Character Recognition\n\n";

    // train the model
//	Ptr<ml::SVM> model = get_model(path_data, split, n_threads);
    // load the model
    Ptr<ml::SVM> model = get_model();

    // drawing window
	namedWindow("Drawing Window", CV_WINDOW_FREERATIO);
	setMouseCallback("Drawing Window", &on_mouse);

	help();

    while(execute){

        imshow("Drawing Window", screenBuffer);
        key = waitKey(10);

        switch ((char)key){
            case 27: // ESC, finish the program
                execute = false;
                break;
            case '+': // increase the radius
            case 'w':
                radius++;
                draw_cursor(last_x, last_y);
                break;
            case '-': // decrease the radius
            case 'q':
                if (radius>1) {
                    radius--;
            		draw_cursor(last_x, last_y);
                }
                break;
            case 'e': // reset the canvas
                imagen = 255;
                draw_cursor(last_x, last_y);
                break;
            case 't': // save the image
                imwrite("output.png", imagen);
                break;
            case 'r': // classify
                classify(model, imagen);
                break;
        }
    }

    destroyWindow("Drawing Window");

    return 0;
}
