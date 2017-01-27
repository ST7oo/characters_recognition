

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

Mat imagen, screenBuffer;
int radius, last_x, last_y, new_size;
bool drawing;
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

void preprocess(Mat img, Mat &row_data, bool debugP = false, int c=0, string n_folder="", Mat mask=Mat(), Mat color=Mat()) {
    Mat image, gray, vect, hist, hsv;
    Mat binary, binary1, result, resized;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    int histSize = 2, black_dots = 0;
    float ratio_bw;
    int channels[] = {2};

    /*cvtColor(color, hsv, CV_BGR2HSV);
    calcHist(&hsv, 1, channels, Mat(), hist, 1, &histSize, 0);
    ratio_bw = hist.at<float>(0)/hist.at<float>(1);
    if(mask.data){
        if (ratio_bw < 1.5 && ratio_bw > 0.02) {
            image = Mat(img.size(),CV_8U, 255);
            img.copyTo(image,mask);
            threshold(image, binary1, 0, 255, THRESH_BINARY | THRESH_OTSU);
        }
        else {
            img.copyTo(image,mask);
            threshold(image, binary1, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
        }
    }
    else {*/
        image = img;
        threshold(image, binary1, 0, 255, THRESH_BINARY | THRESH_OTSU);
    //}

    resize(binary1, resized, Size(new_size,new_size), 0, 0, INTER_NEAREST);

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

    /*if(c%20==0) {
        cout<<ratio_bw<<" - "<<black_dots<<endl;
        imshow("source", img);
        imshow("image", image);
        imshow("binary1", binary1);
        imshow("mask", mask);
        imshow("resized", resized);
        imshow("final", result);
        waitKey();
    }*/
    /*findContours(binary1, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    double areaMax = 0;
    int index;

    if (debugP)  cout<< "countors number: " << contours.size() << endl;

    for( int i = 1; i < (int)contours.size(); i++ ){
        approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        double area1 = contourArea(contours_poly[i]);
        if (debugP) cout <<" area contours["<<i<<"]: "<< area1 << endl;
        if (area1 >areaMax) {
            areaMax = area1;
            index = i;
        }
    }

    if(!binary(boundRect[index]).data){
        cout<<"no data "+to_string(c)<<endl;
    }
    if(boundRect[index].height > 2 && boundRect[index].width > 2){
        result = binary(boundRect[index]);
    }
    else {
        result = binary;
    }
    if(debugP){
        cout<<boundRect[index]<<endl;
        cout<<image.size()<<endl;
        cout<<result.size()<<endl;
    }
    resize(result, resized, Size(new_size, new_size) );

    if (debugP){
        namedWindow( "image resized and normalized", CV_WINDOW_FREERATIO );
        imshow( "image resized and normalized", resized );
        waitKey();
    }*/
    result.convertTo(vect, CV_32F);
    row_data = vect.reshape(1,1);
    /*for (int i=0; i<row_data.cols; i++){
        if(row_data.at<float>(i) < 80){
            row_data.at<float>(i) = 0;
        }
        else if(row_data.at<float>(i) < 140){
            row_data.at<float>(i) = 127;
        }
        else{
            row_data.at<float>(i) = 255;
        }
    }*/
    /*if(c%500==0) {
//                        cout<<row_data<<endl;
        Mat reshaped = row_data.reshape(1,new_size);
        reshaped.convertTo(reshaped, gray.type());
//                        cout<<reshaped<<endl;
        imshow(n_folder+to_string(c),reshaped);
    }*/
    divide(row_data, 255, row_data);
}

int read_folder(string folder, vector<Mat> &data, string n_folder) {
    string name, folder_masks;
    Mat image, row_data, mask;
    DIR *dir;
    struct dirent *ent;
    int c = 0;
    if ((dir = opendir(folder.c_str())) != NULL) {
        if(n_folder.substr(0, 3)=="img"){
            folder_masks = folder;
            folder_masks.replace(folder_masks.find("Img"), 3, "Msk");
        }
        while ((ent = readdir(dir)) != NULL) {
            name = ent->d_name;
            if (name.find(".png")!=string::npos){
                image = imread(folder+name, IMREAD_GRAYSCALE);
                Mat image2 = imread(folder+name);
                mask = imread(folder_masks+name, IMREAD_GRAYSCALE);
                if(image.data) {
                    preprocess(image, row_data, false, c, n_folder, mask, image2);
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

void load_and_preprocess(string path, int n_folder, vector<Mat> &data, int &num_samples){
    string folder;
    int c = 0;

    folder = path+"Fnt/Sample"+fill_zeros(n_folder+1, 3)+"/";
    c += read_folder(folder, data, "fnt"+to_string(n_folder));

    folder = path+"Hnd/Sample"+fill_zeros(n_folder+1, 3)+"/";
    c += read_folder(folder, data, "hnd"+to_string(n_folder));

    folder = path+"Img/Sample"+fill_zeros(n_folder+1, 3)+"/";
//    c += read_folder(folder, data, "img"+to_string(n_folder));

    num_samples = c;
}

// reads data and stores in trainX, trainY, testX, testY
void read_data(string path, Mat &trainX, Mat &testX, Mat &trainY, Mat &testY, int num_classes, float split, int n_threads = 8){
    int train_samples, test_samples;
    vector<thread> threads(n_threads);
    vector<vector<Mat>> data(num_classes, vector<Mat>());
    vector<int> num_samples(num_classes);

    cout<<"Loading images and preprocessing..."<<endl;
    int i = 0, c = 0;
    while (i<num_classes) {
        c = 0;
        for(int j=0; j<n_threads && i<num_classes; j++) {
            threads[j] = thread(load_and_preprocess, path, i+from_class, ref(data[i]), ref(num_samples[i]));
            i++;
            c++;
        }
        for(int j=0; j<c; j++) {
            threads[j].join();
        }
    }
//    load_and_preprocess(path, i+from_class, ref(data[i]), ref(num_samples[i]));
//    i++;
//    load_and_preprocess(path, i+from_class, ref(data[i]), ref(num_samples[i]));
//    i++;
//    load_and_preprocess(path, i+from_class, ref(data[i]), ref(num_samples[i]));
//    waitKey();

    cout<<"Preparing data..."<<endl;
    Mat trainData;
	Mat trainClasses;
    Mat testData;
	Mat testClasses;
	cout<<"Training samples: ";
    for(int i = 0; i < num_classes; i++) {
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
    for(int i = 0; i < num_classes; i++) {
        cout<<num_samples[i] - (int)round(num_samples[i]*split)<<", ";
    }
    cout<<"= "<<testData.rows<<endl;
    trainX = trainData;
    trainY = trainClasses;
    testX = testData;
    testY = testClasses;
}

Ptr<ml::SVM> trainSVM(Mat X, Mat Y) {
    Y.convertTo(Y,CV_32S);
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setC(0.08);
//    svm->setGamma(5.4);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    Ptr<ml::TrainData> td = ml::TrainData::create(X, ml::ROW_SAMPLE, Y);
    svm->train(td);
//    ml::ParamGrid gridC(1e-5,50,5);
//    ml::ParamGrid gridGamma(1e-5,100,5);
//    svm->trainAuto(td, 5);
    cout<<to_string(svm->getC())<<endl;
    cout<<to_string(svm->getGamma())<<endl;
    svm->save("svm.yml");
    return svm;
}

// evaluates the prediction
float evaluate(cv::Mat& predicted, cv::Mat& actual) {
//    cout<<predicted.rows<<" - "<<actual.rows<<endl;
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

void predictSVM(Ptr<ml::SVM> model, Mat X, Mat Y) {
//    cout<<Y.rows<<endl;
    for(int i = 0; i < 9000 && i < Y.rows; i+=30) {
        Mat res;
        Mat sample = X.row(i);
        model->predict(sample, res);
        string actu = get_label(Y.at<float>(i));
        string pred = get_label(res.at<float>(0));
        cout << actu << ":" << pred << "\t";
        /*if (actu != pred) {
            Mat reshaped = X.row(i).reshape(1,new_size);
            reshaped.convertTo(reshaped, 0);
            imshow("test"+to_string(i), reshaped);
        }*/
    }
    cout<<endl;

    Mat predicted(Y.rows, 1, CV_32F);
    model->predict(X, predicted);
    cout << "Accuracy = " << evaluate(predicted, Y) << endl;
}

// calculates the elapsed time
string elapsed_time(std::chrono::steady_clock::time_point start_time, std::chrono::steady_clock::time_point end_time) {
    float elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) / 1000000.0;
    if (elapsed > 60) {
        elapsed /= 60;
        return to_string(elapsed) + " min";
    } else {
        return to_string(elapsed) + " sec";
    }
}

void classify(Ptr<ml::SVM> model, Mat sample) {
    Mat res, row_data;
    preprocess(sample, row_data, false);
    model->predict(row_data, res);
    cout<<get_label(res.at<float>(0))<<endl;
}

Ptr<ml::SVM> get_model(bool train) {

    Ptr<ml::SVM> model;
    int n_classes = 10;
    float split = 0.8;
    int n_threads = 8;

    if (train) {
        string path_data = "data/";
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point end_time;
        Mat trainX, testX, trainY, testY;

        // Reading
        cout << "\nStart reading:\n";

        start_time = std::chrono::steady_clock::now();
        read_data(path_data, trainX, testX, trainY, testY, n_classes, split, n_threads);
        cout<<trainX.size()<<endl;
    //    cout<<trainY.size()<<endl;
        cout<<testX.size()<<endl;
    //    cout<<testY.size()<<endl;
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
    }
    else {
        model = ml::StatModel::load<ml::SVM>("svm.yml");
    }

    return model;
}

void draw_cursor(int x, int y){
	screenBuffer=imagen.clone();
	circle(screenBuffer, Point(x,y), radius, CV_RGB(0,0,0), -1, CV_AA);
}

void draw(int x,int y){
	circle(imagen, Point(x,y), radius, CV_RGB(0,0,0), -1, CV_AA);
	screenBuffer = imagen.clone();
	imshow("Drawing Window", screenBuffer);
}

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

int main()
{
    cout << "CHARS74K\n\n";

	int key;
	bool execute = true;
	bool train = true;
    drawing = false;
    new_size = 16;
	radius = 6;
	last_x=last_y=0;
	imagen= Mat(Size(128,128),CV_8U, 255);
	screenBuffer=imagen.clone();

	Ptr<ml::SVM> model = get_model(train);

	namedWindow("Drawing Window", CV_WINDOW_FREERATIO);
	setMouseCallback("Drawing Window", &on_mouse);

    while(execute){

        imshow("Drawing Window", screenBuffer);
        key = waitKey(10);

        switch ((char)key){
            case 27:
                execute = false;
                break;
            case '+':
            case 'a':
                radius++;
                draw_cursor(last_x, last_y);
                break;
            case '-':
            case 's':
                if (radius>1) {
                    radius--;
            		draw_cursor(last_x, last_y);
                }
                break;
            case 'r':
                imagen = 255;
                draw_cursor(last_x, last_y);
                break;
            case 't':
                imwrite("output.png", imagen);
                break;
            case 'e':
                classify(model, imagen);
                break;
        }
    }

    destroyWindow("Drawing Window");

    return 0;
}
