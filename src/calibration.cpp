#include "../include/calibration.hpp"


bool calibrate() {
  // Defining the dimensions of checkerboard
  int CHECKERBOARD[2]{6, 9};

  // Creating vector to store vectors of 3D points for each checkerboard image
  std::vector<std::vector<cv::Point3f>> objpoints;

  // Creating vector to store vectors of 2D points for each checkerboard image
  std::vector<std::vector<cv::Point2f>> imgpoints;

  // Defining the world coordinates for 3D points
  std::vector<cv::Point3f> objp;
  for (int i{0}; i < CHECKERBOARD[1]; i++) {
    for (int j{0}; j < CHECKERBOARD[0]; j++)
      objp.push_back(cv::Point3f(j, i, 0));
  }

  // Extracting path of individual image stored in a given directory
  std::vector<cv::String> images;
  // Path of the folder containing checkerboard images
  std::string path = "../images/2023*.jpg";

  cv::glob(path, images);

  cv::Mat frame, gray;
  // vector to store the pixel coordinates of detected checker board corners
  std::vector<cv::Point2f> corner_pts;

  // Looping over all the images in the directory
  for (int i{0}; i < images.size(); i++) {
    frame = cv::imread(images[i]);
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Finding checker board corners
    // If desired number of corners are found in the image then success = true
    bool success = cv::findChessboardCorners(
        gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK |
            cv::CALIB_CB_NORMALIZE_IMAGE);

    /*
     * If desired number of corner are detected,
     * we refine the pixel coordinates and display
     * them on the images of checker board
     */
    if (success) {
      cv::TermCriteria criteria(
          cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

      // refining pixel coordinates for given 2d points.
      cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1),
                       criteria);

      // Displaying the detected corner points on the checker board
      cv::drawChessboardCorners(frame,
                                cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]),
                                corner_pts, success);

      objpoints.push_back(objp);
      imgpoints.push_back(corner_pts);
    }
    else{
      std::cerr << "Chessboard corners are not found on picture " << i << std::endl;
      return false;
    }

    cv::imshow("Image", frame);
    cv::waitKey(0);
  }

  cv::destroyAllWindows();

  cv::Mat cameraMatrix, distCoeffs, R, T;

  /*
   * Performing camera calibration by
   * passing the value of known 3D points (objpoints)
   * and corresponding pixel coordinates of the
   * detected corners (imgpoints)
   */
  float error =
      cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols),
                          cameraMatrix, distCoeffs, R, T);

  std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
  std::cout << "distCoeffs : " << distCoeffs << std::endl;
  std::cout << "Error : " << error << std::endl;
  std::cout << "Rotation vector : " << R << std::endl;
  std::cout << "Translation vector : " << T << std::endl;

  // Precompute lens correction interpolation
  cv::Mat mapX, mapY;
  cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Matx33f::eye(),
                              cameraMatrix, cv::Size(gray.cols, gray.rows),
                              CV_32FC1, mapX, mapY);

  // Show lens corrected images
  for (auto const &image : images) {
    cv::Mat img = cv::imread(image, cv::IMREAD_COLOR);
    cv::Mat imgUndistorted;

    // Remap the image using the precomputed interpolation maps.
    cv::remap(img, imgUndistorted, mapX, mapY, cv::INTER_LINEAR);

    // Display
    cv::imshow("Undistorted image", imgUndistorted);
    cv::waitKey(0);
  }

  std::string filename = "../calibration_data/CalibrationData.xml";
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);

  // Write calibration data to xml file
  fs << "R" << R;
  fs << "T" << T;

  fs << "cameraMatrix" << cameraMatrix;
  fs << "distCoeffs" << distCoeffs;

 // Reading the mapping values for stereo image rectification
  cv::FileStorage cv_file2 =
      cv::FileStorage("../calibration_data/stereo_rectify_maps.xml", cv::FileStorage::APPEND);
  // cv_file2 << "Left_Stereo_Map_x" << mapX;
  // cv_file2 << "Left_Stereo_Map_y" << mapY;
  cv_file2<<"Right_Stereo_Map_x" << mapX;
  cv_file2<<"Right_Stereo_Map_y" << mapY;
  cv_file2.release();

  // // To read them back
  // fs.open(filename, cv::FileStorage::READ);
  // fs["R"] >> R;
  // fs["T"] >> T;

  fs.release();
  return true;
}