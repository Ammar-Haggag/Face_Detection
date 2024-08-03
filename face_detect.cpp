#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>

int main() {
    // Load the pre-trained face detection model (Haar Cascade)
    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load(cv::samples::findFile("haarcascade_frontalface_default.xml"))) {
        std::cerr << "Error loading face cascade\n";
        return -1;
    }

    // Load the image
    cv::Mat image = cv::imread("/home/pi/image.jpeg");
    if (image.empty()) {
        std::cerr << "Error loading image\n";
        return -1;
    }

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    // Detect faces
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(gray, faces);

    // Draw rectangles around the detected faces
    for (size_t i = 0; i < faces.size(); i++) {
        cv::rectangle(image, faces[i], cv::Scalar(255, 0, 0), 2);
    }

    // Display the result
    cv::imshow("Detected Faces", image);
    cv::waitKey(0);

    return 0;
}
