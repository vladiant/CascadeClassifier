#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

int main(int argc, char** argv) {
  cv::String keys =
      "{help h usage ? || print this message}"
      "{descriptor d   || cascade descriptor XML file}"
      "{image i        || image filename}";

  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Cacscade classifier object detection sample");

  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }

  const auto image_name = parser.get<std::string>("image");

  cv::Mat img = cv::imread(image_name, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cout << "Failed to load file: " << image_name << '\n';
    return EXIT_FAILURE;
  }

  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

  const auto cascade_file = parser.get<std::string>("descriptor");
  if (!std::filesystem::exists(cascade_file)) {
    std::cout << "Cascade descriptor file not found: " << cascade_file << '\n';
    return EXIT_FAILURE;
  }

  cv::CascadeClassifier object_cascade(cascade_file);

  std::vector<cv::Rect> objects;
  object_cascade.detectMultiScale(gray, objects, 4, 50);

  for (const auto& object : objects) {
    cv::rectangle(img, object, cv::Scalar(255, 255, 0), 2);
  }

  constexpr auto kWinName = "img";
  cv::namedWindow(kWinName, cv::WINDOW_NORMAL);
  cv::imshow(kWinName, img);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}