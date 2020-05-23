#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace cv;
using namespace dnn;
using namespace saliency;
using namespace std;

void compute(Mat frame);
std::vector<Mat> segment(Mat src, Mat rgb);
Mat KMeans(Mat src, int clusterCount);
void findContours(Mat, Mat);

//dnn functions
void classifyImage(Mat frame);
vector<String> getOutputsNames(const Net& net);
void postprocess(Mat& frame, const vector<Mat>& outs);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);


int main() {
    VideoCapture cap("/Users/yudhiesh/Downloads/VehicleFrame.mp4");
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS) / 5;
    cout << "fps is: " << fps << std::endl;
    VideoWriter video("output.mp4v", VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, Size(frame_width, frame_height));

    while (true)
    {
        Mat frame;
        //Skip 10 frames
        for (int i = 0; i < 5; i++)
            cap >> frame;

        if (frame.empty())
            break;

        compute(frame);
        video.write(frame);
        imshow("frame", frame);
        char c = (char)waitKey(10);
        if (c == 27)
            break;
    }
    cap.release();
    video.release();
    destroyAllWindows();
}

void compute(Mat frame) {
    Mat toProcess = frame.clone();
    medianBlur(toProcess, toProcess, 3);

    Ptr<StaticSaliencyFineGrained> salFG = StaticSaliencyFineGrained::create();
    Mat mapSR, mapFG;

    salFG->computeSaliency(toProcess, mapFG);
    mapFG.convertTo(mapFG, CV_8U, 255);

    cvtColor(mapFG, mapFG, COLOR_GRAY2BGR);
    vector<Mat> croppedImages = segment(KMeans(mapFG, 5), frame);

    for (Mat img : croppedImages) {
        classifyImage(img);
    }

}

Mat KMeans(Mat src, int clusterCount) {
    Mat samples(src.rows * src.cols, 3, CV_32F);
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
            for (int z = 0; z < 3; z++)
                samples.at<float>(y + x * src.rows, z) = src.at<Vec3b>(y, x)[z];

    Mat labels;
    int attempts = 5;
    Mat centers;
    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

    Mat dst = Mat::zeros(src.size(), src.type());
    Vec2i pointVal = { 0, 0 };

    //Get color with highest intensity
    for (int y = 0; y < centers.rows; y++) {
        int sum = 0;
        for (int x = 0; x < centers.cols; x++) {
            sum += centers.at<float>(y, x);
        }
        if (sum / 3 > pointVal[1]) {
            pointVal[0] = y;
            pointVal[1] = sum / 3;
        }
    }

    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
        {

            int cluster_idx = labels.at<int>(y + x * src.rows, 0);
            if (cluster_idx == pointVal[0]) {
                dst.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
                dst.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
                dst.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
            }
        }

    cvtColor(dst, dst, COLOR_BGR2GRAY);
    return dst;
}

vector<Mat> segment(Mat src, Mat ori) {
    vector<std::vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Mat> croppedImg;

    threshold(src, src, 0, 255, THRESH_BINARY | THRESH_OTSU);

    int erosion_size = 5;
    erode(src, src,
        getStructuringElement(MORPH_RECT,
            Size(erosion_size * 2 + 1, 1),
            Point(erosion_size, 0))
    );

    int dilation_size = 10;
    dilate(src, src,
        getStructuringElement(MORPH_RECT,
            Size(1, dilation_size * 2 + 1),
            Point(0, dilation_size))
    );

    findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    Mat drawing = Mat::zeros(src.size(), CV_8UC3);
    // Original image clone
    RNG rng(12345);

    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        Rect r = boundingRect(contours.at(i));
        double ratio = r.width / r.height;

        if (r.width < ori.cols * 0.05 || r.height < ori.rows * 0.1 || r.y < ori.rows * 0.25 || (r.x < ori.cols * 0.3) || ratio > 3.0) {
            continue;
        }

        croppedImg.push_back(ori(r));
    }
    return croppedImg;
}


//dnn variables
vector<std::string> classes;

void classifyImage(Mat frame) {
    // Initialize the parameters
//    float confThreshold = 0.5; // Confidence threshold
//    float nmsThreshold = 0.4;  // Non-maximum suppression threshold
    int inpWidth = 416;        // Width of network's input image
    int inpHeight = 416;       // Height of network's input image

    // Load names of classes
    string classesFile = "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;

    while (getline(ifs, line)) classes.push_back(line);

    // Give the configuration and weight files for the model
    String modelConfiguration = "/Users/yudhiesh/darknet/cfg/yolov3.cfg";
    String modelWeights = "/Users/yudhiesh/darknet/yolov3.weights";

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Open the image file
    string str = "Test.jpg";
    //std::ifstream ifile(str);
    //if (!ifile) throw("error");
    str.replace(str.end() - 4, str.end(), "_yolo_out.jpg");
    string outputFile = str;

    // Create a 4D blob from a frame.
    Mat blob;
    blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

    //Set the input to the network
    net.setInput(blob);

    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Remove the bounding boxes with low confidence
    postprocess(frame, outs);

}

vector<String> getOutputsNames(const Net& net)
{
    static std::vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void postprocess(Mat& frame, const std::vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    float confThreshold = 0.5;
    float nmsThreshold = 0.4;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
            box.x + box.width, box.y + box.height, frame);
    }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255));

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 1, 2, &baseLine);
    top = max(top, labelSize.height);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255));
    
}
