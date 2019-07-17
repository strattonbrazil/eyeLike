#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

/* Attempt at supporting openCV version 4.0.1 or higher */
#if CV_MAJOR_VERSION >= 4
#define CV_WINDOW_NORMAL                cv::WINDOW_NORMAL
#define CV_BGR2YCrCb                    cv::COLOR_BGR2YCrCb
#define CV_HAAR_SCALE_IMAGE             cv::CASCADE_SCALE_IMAGE
#define CV_HAAR_FIND_BIGGEST_OBJECT     cv::CASCADE_FIND_BIGGEST_OBJECT
#endif


/** Constants **/
const int MAX_QUEUED_POINTS = 1000;
const int POINT_SEARCH_RANGE = 20;

/** Function Headers */
void detectAndDisplay( cv::Mat frame );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String face_cascade_name = "../res/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);
std::deque<cv::Point2i> leftRelativePoints; // relative to eye region
std::deque<cv::Point2i> rightRelativePoints; // relative to eye region
std::deque<cv::Point2f> normalPoints; // normalized

enum POSITIONS {
  LEFT,
  RIGHT,
  CENTER,
  UP,
  DOWN
};

int region(cv::Point2f p) {
  if (p.x < -0.9) {
    return RIGHT;
  } else if (p.x > 0.9) {
    return LEFT;
  } else if (p.y < -0.9) {
    return UP;
  } else if (p.y > 0.9) {
    return DOWN;
  } else {
    return CENTER;
  }
}

long getEpochMilliseconds() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

bool hasFiredEvent = false;
long lastEventMs;
bool canFireEvent() {
  if (!hasFiredEvent)
    return true;

  long currentTime = getEpochMilliseconds();
  return currentTime - lastEventMs > 1000; // it's been more than X milliseconds since last event
}

void fireEvent(int region) {
    if (region == LEFT) {
      std::cout << "go left" << std::endl;
    } else if (region == RIGHT) {
      std::cout << "go right!" << std::endl;
    } else if (region == UP) {
      std::cout << "go up!" << std::endl;
    } else if (region == DOWN) {
      std::cout << "go down!" << std::endl;
    }

  // save off for thresholding
  lastEventMs = getEpochMilliseconds();
  hasFiredEvent = true;
}

void findKeyEvent(std::deque<cv::Point2f> points)
{
  if (points.size() > POINT_SEARCH_RANGE) {
    const int HALF_SEARCH_RANGE = POINT_SEARCH_RANGE / 2;
    //cv::Point2f firstPoint = points[0];
    //cv::Point2f midPoint = points[points.size() / 2];
    //cv::Point2f lastPoint = points[points.size() - 1];

    int firstRegion = region(points[points.size() - POINT_SEARCH_RANGE]);
    int midRegion = region(points[points.size() - HALF_SEARCH_RANGE]);
    int lastRegion = region(points[points.size() - 1]);

    // process direction
    if (midRegion != CENTER) {
      if (canFireEvent()) {
        fireEvent(midRegion);


      }
    }

    /*
    */
    // region 
  }
  // for (int i = std::max(0, (int)(points.size() - MAX_QUEUED_POINTS)); i < points.size(); i++) {

  // }
}

inline bool isInteger(const std::string & s)
{
   if(s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+'))) return false;

   char * p;
   strtol(s.c_str(), &p, 10);

   return (*p == 0);
}

/**
 * @function main
 */
int main( int argc, const char** argv ) {
  cv::Mat frame;

  // Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };
  cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(main_window_name, 400, 100);
  cv::namedWindow(face_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(face_window_name, 10, 100);
  cv::namedWindow("Right Eye",CV_WINDOW_NORMAL);
  cv::moveWindow("Right Eye", 10, 600);
  cv::namedWindow("Left Eye",CV_WINDOW_NORMAL);
  cv::moveWindow("Left Eye", 10, 800);

  /* As the matrix dichotomy will not be applied, these windows are useless.
  cv::namedWindow("aa",CV_WINDOW_NORMAL);
  cv::moveWindow("aa", 10, 800);
  cv::namedWindow("aaa",CV_WINDOW_NORMAL);
  cv::moveWindow("aaa", 10, 800);*/

  createCornerKernels();
  ellipse(skinCrCbHist, cv::Point(113, 155), cv::Size(23, 15),
          43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);

  // I make an attempt at supporting both 2.x and 3.x OpenCV
#if CV_MAJOR_VERSION < 3
  CvCapture* capture = cvCaptureFromCAM( 0 );
  if( capture ) {
    while( true ) {
      frame = cvQueryFrame( capture );
#else
  cv::VideoCapture capture;
  if (argc < 2) {
    std::cerr << "missing parameter for webcam or file" << std::endl;
    exit(EXIT_FAILURE);
  }
  else if (isInteger(argv[1])) {
    capture = cv::VideoCapture(atoi(argv[1]));
  } else {
    capture = cv::VideoCapture(argv[1]);
  }
  ("/tmp/hero.mp4");
  if( capture.isOpened() ) {
    while( true ) {
      capture.read(frame);
#endif
      // mirror it
      cv::flip(frame, frame, 1);
      frame.copyTo(debugImage);

      // Apply the classifier to the frame
      if( !frame.empty() ) {
        detectAndDisplay( frame );
      }
      else {
        printf(" --(!) No captured frame -- Break!");
        break;
      }

      //imshow(main_window_name,debugImage);

      int c = cv::waitKey(10);
      if( (char)c == 'c' ) { break; }
      if( (char)c == 'f' ) {
        imwrite("frame.png",frame);
      }

    }
  }

  releaseCornerKernels();

  return 0;
}

void findEyes(cv::Mat frame_gray, cv::Rect face) {
  cv::Mat faceROI = frame_gray(face);
  cv::Mat debugFace = faceROI;

  if (kSmoothFaceImage) {
    double sigma = kSmoothFaceFactor * face.width;
    GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
  }
  //-- Find eye regions and draw them
  int eye_region_width = face.width * (kEyePercentWidth/100.0);
  int eye_region_height = face.width * (kEyePercentHeight/100.0);
  int eye_region_top = face.height * (kEyePercentTop/100.0);
  cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
                         eye_region_top,eye_region_width,eye_region_height);
  cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
                          eye_region_top,eye_region_width,eye_region_height);

  //-- Find Eye Centers
  cv::Point leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
  cv::Point rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");
  // get corner regions
  cv::Rect leftRightCornerRegion(leftEyeRegion);
  leftRightCornerRegion.width -= leftPupil.x;
  leftRightCornerRegion.x += leftPupil.x;
  leftRightCornerRegion.height /= 2;
  leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
  cv::Rect leftLeftCornerRegion(leftEyeRegion);
  leftLeftCornerRegion.width = leftPupil.x;
  leftLeftCornerRegion.height /= 2;
  leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
  cv::Rect rightLeftCornerRegion(rightEyeRegion);
  rightLeftCornerRegion.width = rightPupil.x;
  rightLeftCornerRegion.height /= 2;
  rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
  cv::Rect rightRightCornerRegion(rightEyeRegion);
  rightRightCornerRegion.width -= rightPupil.x;
  rightRightCornerRegion.x += rightPupil.x;
  rightRightCornerRegion.height /= 2;
  rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
  rectangle(debugFace,leftRightCornerRegion,200);
  rectangle(debugFace,leftLeftCornerRegion,200);
  rectangle(debugFace,rightLeftCornerRegion,200);
  rectangle(debugFace,rightRightCornerRegion,200);
  // change eye centers to face coordinates
  rightPupil.x += rightEyeRegion.x;
  rightPupil.y += rightEyeRegion.y;
  leftPupil.x += leftEyeRegion.x;
  leftPupil.y += leftEyeRegion.y;
  // draw eye centers
  circle(debugFace, rightPupil, 3, 1234);
  circle(debugFace, leftPupil, 3, 1234);

  leftRelativePoints.push_back(leftPupil);
  rightRelativePoints.push_back(rightPupil);

  if (leftRelativePoints.size() > 24) {
    // these are magic ratios of where the pupils fall in the boxes
    cv::Point2f eyeCenter(0.5 * (leftEyeRegion.x + rightEyeRegion.x + rightEyeRegion.width),
                          leftLeftCornerRegion.y + 0.6 * leftLeftCornerRegion.height);

    cv::Point2f currentRelativeCenter((leftPupil.x + rightPupil.x) * 0.5, 
                              (leftPupil.y + rightPupil.y) * 0.5);

    const float pixelsToNormal = leftLeftCornerRegion.height / 4.0f;
    cv::Point2f currentNormalCenter((currentRelativeCenter.x - eyeCenter.x) / pixelsToNormal,
                                    (currentRelativeCenter.y - eyeCenter.y) / pixelsToNormal);
    //std::cout << currentNormalCenter << std::endl;                                    
    // cv::Point2f leftNormalPoint = cv::Point2f((leftPupil.x - avgLeft.x) / EYE_TO_PUPIL,
    //                                             (leftPupil.y - avgLeft.y) / EYE_TO_PUPIL);
    normalPoints.push_back(currentNormalCenter);
  //  std::cout << leftNormalPoint << std::endl;

    findKeyEvent(normalPoints);

    // make sure queues don't grow too big
    while (leftRelativePoints.size() > MAX_QUEUED_POINTS) {
      leftRelativePoints.pop_front();
    }
    while (rightRelativePoints.size() > MAX_QUEUED_POINTS) {
      rightRelativePoints.pop_front();
    }
    while (normalPoints.size() > MAX_QUEUED_POINTS) {
      normalPoints.pop_front();
    }

    // draw average left eye position
    circle(debugFace, eyeCenter, 5, 1234, 2);
    circle(debugFace, currentRelativeCenter, 3, 234);
  }

  //-- Find Eye Corners
  if (kEnableEyeCorner) {
    cv::Point2f leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
    leftRightCorner.x += leftRightCornerRegion.x;
    leftRightCorner.y += leftRightCornerRegion.y;
    cv::Point2f leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
    leftLeftCorner.x += leftLeftCornerRegion.x;
    leftLeftCorner.y += leftLeftCornerRegion.y;
    cv::Point2f rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
    rightLeftCorner.x += rightLeftCornerRegion.x;
    rightLeftCorner.y += rightLeftCornerRegion.y;
    cv::Point2f rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
    rightRightCorner.x += rightRightCornerRegion.x;
    rightRightCorner.y += rightRightCornerRegion.y;
    circle(faceROI, leftRightCorner, 3, 200);
    circle(faceROI, leftLeftCorner, 3, 200);
    circle(faceROI, rightLeftCorner, 3, 200);
    circle(faceROI, rightRightCorner, 3, 200);
  }

  imshow(face_window_name, faceROI);
//  cv::Rect roi( cv::Point( 0, 0 ), faceROI.size());
//  cv::Mat destinationROI = debugImage( roi );
//  faceROI.copyTo( destinationROI );
}


cv::Mat findSkin (cv::Mat &frame) {
  cv::Mat input;
  cv::Mat output = cv::Mat(frame.rows,frame.cols, CV_8U);

  cvtColor(frame, input, CV_BGR2YCrCb);

  for (int y = 0; y < input.rows; ++y) {
    const cv::Vec3b *Mr = input.ptr<cv::Vec3b>(y);
//    uchar *Or = output.ptr<uchar>(y);
    cv::Vec3b *Or = frame.ptr<cv::Vec3b>(y);
    for (int x = 0; x < input.cols; ++x) {
      cv::Vec3b ycrcb = Mr[x];
//      Or[x] = (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0) ? 255 : 0;
      if(skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) {
        Or[x] = cv::Vec3b(0,0,0);
      }
    }
  }
  return output;
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( cv::Mat frame ) {
  std::vector<cv::Rect> faces;
  //cv::Mat frame_gray;

  std::vector<cv::Mat> rgbChannels(3);
  cv::split(frame, rgbChannels);
  cv::Mat frame_gray = rgbChannels[2];

  //cvtColor( frame, frame_gray, CV_BGR2GRAY );
  //equalizeHist( frame_gray, frame_gray );
  //cv::pow(frame_gray, CV_64F, frame_gray);
  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
//  findSkin(debugImage);

  for( int i = 0; i < faces.size(); i++ )
  {
    rectangle(debugImage, faces[i], 1234);
  }
  //-- Show what you got
  if (faces.size() > 0) {
    findEyes(frame_gray, faces[0]);
  }
}
