#define _CRT_SECURE_NO_WARNINGS
#include "functions.h"
#include "ipaConfig.h"
#include "ucasConfig.h"
#include "ucasImageUtils.h"
#include "ucasMathUtils.h"
#include "ucasLog.h"
#include "ucasTypes.h"
#include "ucasStringUtils.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <sys/stat.h>
struct Fracture {
    int regionId;
    std::string method;
    cv::Point location;
    double score;
    cv::Rect box;
    Fracture()
        : regionId(0),
        method(""),
        location(0, 0),
        score(0.0),
        box() {
    }
};
struct InternalGradientDebug {
    int id;
    cv::Rect roi;
    cv::Point center;
    double meanGray;
    double stdGray;
    double meanGradient;
    double stdGradient;
    double maxGradient;
    double darkDifference;
    bool possibleFracture;
    InternalGradientDebug()
        : id(0),
        roi(),
        center(0, 0),
        meanGray(0.0),
        stdGray(0.0),
        meanGradient(0.0),
        stdGradient(0.0),
        maxGradient(0.0),
        darkDifference(0.0),
        possibleFracture(false) {
    }
};
struct FractureCluster {
    std::vector<Fracture> items;
};
struct ClusterRepresentative {
    Fracture fracture;
    double clusterScore;
    int clusterSize;
    ClusterRepresentative()
        : fracture(),
        clusterScore(0.0),
        clusterSize(0) {
    }
};
struct InternalAdaptiveThresholds {
    double meanGray;
    double meanGradient;
    double stdGradient;
    double maxGradient;
    double stdGray;
    double darkDifference;
    InternalAdaptiveThresholds()
        : meanGray(0.0),
        meanGradient(0.0),
        stdGradient(0.0),
        maxGradient(0.0),
        stdGray(0.0),
        darkDifference(0.0) {
    }
};
struct InternalDecisionDebug {
    bool passMeanGray;
    bool passMeanGradient;
    bool passStdGradient;
    bool passMaxGradient;
    bool passStdGray;
    bool passDarkDifference;
    int strongTests;
    bool veryStrongGradient;
    bool balancedStrong;
    bool possibleFracture;
    InternalDecisionDebug()
        : passMeanGray(false),
        passMeanGradient(false),
        passStdGradient(false),
        passMaxGradient(false),
        passStdGray(false),
        passDarkDifference(false),
        strongTests(0),
        veryStrongGradient(false),
        balancedStrong(false),
        possibleFracture(false) {
    }
};
struct GroundTruthBox {
    int classId;
    cv::Rect box;
    bool matched;
    GroundTruthBox()
        : classId(-1),
        box(),
        matched(false) {
    }
};
struct EvaluationStats {
    int tp;
    int fp;
    int fn;
    EvaluationStats()
        : tp(0),
        fp(0),
        fn(0) {
    }
};
static int MIN_CONTOUR_SIZE = 40;
static int EXTERNAL_MIN_DISTANCE_BETWEEN_FRACTURES = 100;
static int INTERNAL_MIN_DISTANCE_BETWEEN_FRACTURES = 90;
static int CLUSTER_RADIUS = 90;
static double CLUSTER_SCORE_WEIGHT = 0.70;
static double CLUSTER_CENTRALITY_WEIGHT = 0.20;
static double CLUSTER_LOCAL_SUPPORT_WEIGHT = 0.10;
static int FRACTURE_BOX_RADIUS_X = 90;
static int FRACTURE_BOX_RADIUS_Y = 90;
static double FRACTURE_BOX_SIZE_FACTOR = 1.8;
static int FRACTURE_BOX_MIN_HEIGHT = 90;
static int FRACTURE_EXTENT_SEARCH_RADIUS = 60;
static double FRACTURE_EXTENT_RELATIVE_THRESHOLD = 0.45;
static int MAX_FRACTURES_TO_SHOW = 100;
static double CLAHE_CLIP = 0.1;
static int BONE_OFFSET = 25;
static int BONE_OPEN_SIZE = 11;
static int BONE_CLOSE_SIZE = 11;
static int CANNY_LOW = 0;
static int CANNY_HIGH = 255;
static double IOU_THRESHOLD = 0.50;
// Nel dataset YOLO la classe 3 corrisponde alla frattura.
static const int FRACTURE_LABEL_CLASS_ID = 3;
// Dataset del progetto: immagini radiografiche e relative label YOLO.
// Le label devono avere lo stesso nome dell'immagine e formato .txt.
static const std::string DEFAULT_IMAGES_DIR =
"/Users/vincenzosauzullo/Desktop/Bone_Fracture_Detection/img_fracture";
static const std::string DEFAULT_LABELS_DIR =
"/Users/vincenzosauzullo/Desktop/Bone_Fracture_Detection/img_fracture/labels";

// Cartella ordinata per i risultati generati dal modulo C++.
// A ogni nuova esecuzione viene cancellata e ricreata automaticamente.
static const std::string DEFAULT_RESULTS_DIR =
"/Users/vincenzosauzullo/Desktop/Bone_Fracture_Detection/IPA/risultati_rilevamento_fratture";

static std::string LABELS_DIR = DEFAULT_LABELS_DIR;
static int CONTOUR_SAMPLE_DIVISOR = 700;
static int DIRECTION_OFFSET = 22;
static double DIRECTION_CHANGE_THRESHOLD = 20.0;
static int EDGE_COUNT_THRESHOLD = 30;
static int EDGE_ROI_RADIUS = 22;
static double EDGE_DENSITY_THRESHOLD = 0.012;
static int BORDER_MARGIN = 20;
static int EXTERNAL_BOX_SHIFT_UP = 16;
static double IGNORE_TOP_RATIO = 0.30;
static double IGNORE_BOTTOM_RATIO = 0.20;
static double DISPLAY_CLAHE_CLIP = 2.5;
static int DISPLAY_CLAHE_TILE = 8;
static double UNSHARP_AMOUNT = 1.4;
static double UNSHARP_BLUR_SIGMA = 1.2;
static int INTERNAL_MARGIN_FROM_CONTOUR = 7;
static int INTERNAL_SCAN_STEP = 10;
static int INTERNAL_ROI_RADIUS = 83;
static int INTERNAL_MIN_MASK_PIXELS = 180;
static double INTERNAL_MEAN_GRAY_MIN_THRESHOLD = 110.0;
static double INTERNAL_GRADIENT_MEAN_MIN_THRESHOLD = 520.0;
static double INTERNAL_GRADIENT_STD_MIN_THRESHOLD = 250.0;
static double INTERNAL_GRADIENT_PEAK_MIN_THRESHOLD = 1500.0;
static double INTERNAL_GRAY_STD_MIN_THRESHOLD = 22.0;
static double INTERNAL_DARK_LINE_MIN_THRESHOLD = 50.0;
static double ADAPTIVE_MEAN_GRADIENT_PERCENTILE = 0.88;
static double ADAPTIVE_STD_GRADIENT_PERCENTILE = 0.33;
static double ADAPTIVE_MAX_GRADIENT_PERCENTILE = 0.90;
static double ADAPTIVE_GRAY_STD_PERCENTILE = 0.62;
static double ADAPTIVE_DARK_DIFF_PERCENTILE = 0.50;
static double INTERNAL_GRADIENT_MEAN_MAX_THRESHOLD = 670.0;
static double INTERNAL_GRADIENT_STD_MAX_THRESHOLD = 430.0;
static double INTERNAL_GRADIENT_PEAK_MAX_THRESHOLD = 3000.0;
static double INTERNAL_GRAY_STD_MAX_THRESHOLD = 38.0;
static double INTERNAL_DARK_LINE_MAX_THRESHOLD = 95.0;
static int INTERNAL_MIN_STRONG_TESTS = 4;
static double INTERNAL_GRADIENT_BLUR_SIGMA = 0.64;
static double LOCAL_GRAY_CONTRAST_WEIGHT = 1.89;
static double SOBEL_SCALE_FACTOR = 4.8;

// Piccolo contesto esterno aggiunto SOLO alle ROI usate per estrarre/salvare le feature.
// La detection, i box originali e il calcolo IoU rimangono invariati.
static double FEATURE_ROI_CONTEXT_RATIO = 0.08;
static int FEATURE_ROI_CONTEXT_MIN_PIXELS = 6;
static int FEATURE_ROI_CONTEXT_MAX_PIXELS = 18;
int makeOdd(int v) {
    if (v < 1) return 1;
    return (v % 2 == 0) ? v + 1 : v;
}
cv::Mat kernel(int size) {
    return cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(makeOdd(size), makeOdd(size))
    );
}
double clampDouble(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}
int clampInt(int v, int lo, int hi) {
    return std::max(lo, std::min(hi, v));
}
cv::Rect clampRectToImage(const cv::Rect& r, const cv::Size& imageSize);
double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    q = clampDouble(q, 0.0, 1.0);
    std::sort(values.begin(), values.end());
    double index = q * static_cast<double>(values.size() - 1);
    int i0 = static_cast<int>(std::floor(index));
    int i1 = static_cast<int>(std::ceil(index));
    if (i0 == i1) return values[i0];
    double t = index - static_cast<double>(i0);
    return values[i0] * (1.0 - t) + values[i1] * t;
}
bool insideAllowedVerticalBand(const cv::Point& p, const cv::Size& imageSize) {
    int topLimit = static_cast<int>(static_cast<double>(imageSize.height) * IGNORE_TOP_RATIO);
    int bottomLimit = static_cast<int>(static_cast<double>(imageSize.height) * (1.0 - IGNORE_BOTTOM_RATIO));
    return p.y >= topLimit && p.y <= bottomLimit;
}
cv::Mat improveImage(const cv::Mat& img, int bitsUsed) {
    if (img.channels() != 1) UCAS_THROW("L'immagine deve essere in scala di grigi");
    cv::Mat out = img.clone();
    int bits = bitsUsed > 0 ? bitsUsed : ucas::imdepth_detect(out);
    if (bits > 8) ucas::imrescale(out, bits, 8);
    if (out.depth() != CV_8U) cv::normalize(out, out, 0, 255, cv::NORM_MINMAX, CV_8U);
    return out;
}
cv::Mat enhanceGrayForDisplay(const cv::Mat& gray, const cv::Mat& boneMask) {
    cv::Mat claheImg;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
        DISPLAY_CLAHE_CLIP,
        cv::Size(DISPLAY_CLAHE_TILE, DISPLAY_CLAHE_TILE)
    );
    clahe->apply(gray, claheImg);
    cv::Mat blurred;
    cv::GaussianBlur(claheImg, blurred, cv::Size(0, 0), UNSHARP_BLUR_SIGMA);
    cv::Mat sharpened;
    cv::addWeighted(claheImg, 1.0 + UNSHARP_AMOUNT, blurred, -UNSHARP_AMOUNT, 0, sharpened);
    cv::Mat enhanced = gray.clone();
    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(sharpened, &minVal, &maxVal, 0, 0, boneMask);
    if (maxVal > minVal) {
        cv::Mat normalized;
        sharpened.convertTo(
            normalized,
            CV_8U,
            255.0 / (maxVal - minVal),
            -minVal * 255.0 / (maxVal - minVal)
        );
        normalized.copyTo(enhanced, boneMask);
    }
    else {
        sharpened.copyTo(enhanced, boneMask);
    }
    return enhanced;
}
int otsu(const cv::Mat& gray) {
    return ucas::getOtsuAutoThreshold(ucas::histogram(gray, 256));
}
int otsuMasked(const cv::Mat& gray, const cv::Mat& mask) {
    std::vector<uchar> pixels;
    for (int y = 0; y < gray.rows; ++y) {
        const uchar* g = gray.ptr<uchar>(y);
        const uchar* m = mask.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x) {
            if (m[x]) pixels.push_back(g[x]);
        }
    }
    if (pixels.empty()) UCAS_THROW("Maschera vuota");
    cv::Mat data(static_cast<int>(pixels.size()), 1, CV_8U, pixels.data());
    return otsu(data);
}
cv::Mat largestComponent(const cv::Mat& mask) {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids);
    if (n <= 1) return cv::Mat::zeros(mask.size(), CV_8U);
    int best = 1;
    for (int i = 2; i < n; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) > stats.at<int>(best, cv::CC_STAT_AREA)) {
            best = i;
        }
    }
    cv::Mat out = cv::Mat::zeros(mask.size(), CV_8U);
    out.setTo(255, labels == best);
    return out;
}
std::vector<cv::Point> largestContour(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) return std::vector<cv::Point>();
    size_t best = 0;
    double bestArea = cv::contourArea(contours[0]);
    for (size_t i = 1; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > bestArea) {
            bestArea = area;
            best = i;
        }
    }
    return contours[best];
}
cv::Rect safeBox(const cv::Point& p, const cv::Size& size, int r) {
    int x1 = std::max(0, p.x - r);
    int y1 = std::max(0, p.y - r);
    int x2 = std::min(size.width, p.x + r);
    int y2 = std::min(size.height, p.y + r);
    return cv::Rect(x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1));
}
int findLeftBoneLimitUntilBlack(const cv::Mat& boneMask, const cv::Point& center, int maxRadiusX) {
    if (center.y < 0 || center.y >= boneMask.rows || center.x < 0 || center.x >= boneMask.cols) {
        return clampInt(center.x - maxRadiusX, 0, std::max(0, boneMask.cols - 1));
    }
    int minX = std::max(0, center.x - maxRadiusX);
    int x = center.x;
    while (x > minX && boneMask.at<uchar>(center.y, x - 1) > 0) {
        --x;
    }
    return x;
}
int findRightBoneLimitUntilBlack(const cv::Mat& boneMask, const cv::Point& center, int maxRadiusX) {
    if (center.y < 0 || center.y >= boneMask.rows || center.x < 0 || center.x >= boneMask.cols) {
        return clampInt(center.x + maxRadiusX, 0, boneMask.cols);
    }
    int maxX = std::min(boneMask.cols - 1, center.x + maxRadiusX);
    int x = center.x;
    while (x < maxX && boneMask.at<uchar>(center.y, x + 1) > 0) {
        ++x;
    }
    return std::min(boneMask.cols, x + 1);
}
cv::Rect estimateFractureExtentFromGradient(
    const cv::Point& center,
    const cv::Mat& gradient,
    const cv::Mat& boneMask,
    int searchRadius,
    double relativeThreshold
) {
    if (gradient.empty() || boneMask.empty()) {
        return cv::Rect(center.x, center.y, 1, 1);
    }
    cv::Rect searchRoi = safeBox(center, gradient.size(), searchRadius);
    cv::Mat roiGradient = gradient(searchRoi);
    cv::Mat roiMask = boneMask(searchRoi);
    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(roiGradient, &minVal, &maxVal, 0, 0, roiMask);
    if (maxVal <= 0.0) {
        return cv::Rect(center.x, center.y, 1, 1);
    }
    cv::Mat strongPixels;
    cv::threshold(roiGradient, strongPixels, maxVal * relativeThreshold, 255, cv::THRESH_BINARY);
    strongPixels.convertTo(strongPixels, CV_8U);
    strongPixels.setTo(0, ~roiMask);
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(strongPixels.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return cv::Rect(center.x, center.y, 1, 1);
    }
    cv::Point localCenter(center.x - searchRoi.x, center.y - searchRoi.y);
    int bestIndex = -1;
    double bestDistance = DBL_MAX;
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::Rect r = cv::boundingRect(contours[i]);
        cv::Point rc(r.x + r.width / 2, r.y + r.height / 2);
        double d = cv::norm(rc - localCenter);
        if (d < bestDistance) {
            bestDistance = d;
            bestIndex = static_cast<int>(i);
        }
    }
    if (bestIndex < 0) {
        return cv::Rect(center.x, center.y, 1, 1);
    }
    cv::Rect localBox = cv::boundingRect(contours[bestIndex]);
    return cv::Rect(
        searchRoi.x + localBox.x,
        searchRoi.y + localBox.y,
        localBox.width,
        localBox.height
    );
}
cv::Rect adaptiveFractureBox(
    const cv::Point& center,
    const cv::Size& imageSize,
    const cv::Mat& boneMask,
    const cv::Mat& gradient
) {
    int leftX = findLeftBoneLimitUntilBlack(boneMask, center, FRACTURE_BOX_RADIUS_X);
    int rightX = findRightBoneLimitUntilBlack(boneMask, center, FRACTURE_BOX_RADIUS_X);
    if (rightX <= leftX) {
        leftX = std::max(0, center.x - FRACTURE_BOX_RADIUS_X);
        rightX = std::min(imageSize.width, center.x + FRACTURE_BOX_RADIUS_X);
    }
    cv::Rect fractureExtent = estimateFractureExtentFromGradient(
        center,
        gradient,
        boneMask,
        FRACTURE_EXTENT_SEARCH_RADIUS,
        FRACTURE_EXTENT_RELATIVE_THRESHOLD
    );
    int halfHeightFromFracture = static_cast<int>(std::round(
        static_cast<double>(fractureExtent.height) * FRACTURE_BOX_SIZE_FACTOR / 2.0
    ));
    int horizontalWidth = std::max(1, rightX - leftX);
    int halfHeightFromHorizontalWidth = static_cast<int>(std::round(
        static_cast<double>(horizontalWidth) * 0.35
    ));
    int desiredRadiusY = std::max(halfHeightFromFracture, halfHeightFromHorizontalWidth);
    int radiusY = clampInt(
        desiredRadiusY,
        FRACTURE_BOX_MIN_HEIGHT / 2,
        FRACTURE_BOX_RADIUS_Y
    );
    int topY = center.y - radiusY;
    int height = radiusY * 2;
    return clampRectToImage(
        cv::Rect(leftX, topY, rightX - leftX, height),
        imageSize
    );
}
cv::Rect shiftBoxUp(const cv::Rect& box, int shiftUp, const cv::Size& imageSize) {
    int safeShift = std::max(0, shiftUp);
    return clampRectToImage(
        cv::Rect(box.x, box.y - safeShift, box.width, box.height),
        imageSize
    );
}
void fillHoles(cv::Mat& mask) {
    cv::Mat tmp;
    cv::copyMakeBorder(mask, tmp, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::floodFill(tmp, cv::Point(0, 0), cv::Scalar(255));
    tmp = tmp(cv::Rect(1, 1, mask.cols, mask.rows));
    cv::Mat holes;
    cv::bitwise_not(tmp, holes);
    mask |= holes;
}
void computeSegmentation(const cv::Mat& gray, cv::Mat& boneMask, cv::Mat& edges) {
    cv::Mat bodyMask = gray.clone();
    ucas::binarize(bodyMask, otsu(gray));
    bodyMask = largestComponent(bodyMask);
    if (cv::countNonZero(bodyMask) == 0) UCAS_THROW("Nessun arto trovato");
    cv::Mat claheImg;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
        std::max(0.1, CLAHE_CLIP),
        cv::Size(8, 8)
    );
    clahe->apply(gray, claheImg);
    int threshold = std::max(0, otsuMasked(claheImg, bodyMask) - BONE_OFFSET);
    boneMask = claheImg.clone();
    ucas::binarize(boneMask, threshold);
    boneMask.setTo(0, ~bodyMask);
    cv::morphologyEx(boneMask, boneMask, cv::MORPH_OPEN, kernel(BONE_OPEN_SIZE));
    cv::morphologyEx(boneMask, boneMask, cv::MORPH_CLOSE, kernel(BONE_CLOSE_SIZE));
    boneMask = largestComponent(boneMask);
    fillHoles(boneMask);
    if (cv::countNonZero(boneMask) == 0) UCAS_THROW("Nessun osso trovato");
    cv::Mat cannyInput = cv::Mat::zeros(gray.size(), gray.type());
    gray.copyTo(cannyInput, boneMask);
    cv::GaussianBlur(cannyInput, cannyInput, cv::Size(0, 0), 1.0);
    int low = std::max(0, std::min(255, CANNY_LOW));
    int high = std::max(low + 1, std::min(255, CANNY_HIGH));
    cv::Canny(cannyInput, edges, low, high);
    edges.setTo(0, ~boneMask);
}
double normalizeAngleDiff(double diff) {
    while (diff > 180.0) diff -= 360.0;
    while (diff < -180.0) diff += 360.0;
    return std::abs(diff);
}
double directionChangeDeg(const cv::Point& a, const cv::Point& b, const cv::Point& c) {
    cv::Point2d v1 = b - a;
    cv::Point2d v2 = c - b;
    if (cv::norm(v1) < 1e-6 || cv::norm(v2) < 1e-6) return 0.0;
    double a1 = std::atan2(v1.y, v1.x) * 180.0 / CV_PI;
    double a2 = std::atan2(v2.y, v2.x) * 180.0 / CV_PI;
    return normalizeAngleDiff(a2 - a1);
}
bool nearBorder(const cv::Point& p, const cv::Rect& box) {
    return std::abs(p.x - box.x) < BORDER_MARGIN ||
        std::abs(p.x - (box.x + box.width)) < BORDER_MARGIN ||
        std::abs(p.y - box.y) < BORDER_MARGIN ||
        std::abs(p.y - (box.y + box.height)) < BORDER_MARGIN;
}
bool validExternalFracture(double directionChange, int edgeCount, double edgeDensity) {
    return directionChange >= DIRECTION_CHANGE_THRESHOLD &&
        edgeCount >= EDGE_COUNT_THRESHOLD &&
        edgeDensity >= EDGE_DENSITY_THRESHOLD;
}
std::vector<Fracture> detectExternalFractures(const cv::Mat& boneMask, const cv::Mat& edges) {
    std::vector<Fracture> candidates;
    std::vector<cv::Point> contour = largestContour(boneMask);
    if (contour.size() < static_cast<size_t>(MIN_CONTOUR_SIZE)) return std::vector<Fracture>();
    cv::Rect boneBox = cv::boundingRect(contour);
    int step = std::max(1, static_cast<int>(contour.size()) / CONTOUR_SAMPLE_DIVISOR);
    int externalId = 1;
    for (int i = 0; i < static_cast<int>(contour.size()); i += step) {
        cv::Point p = contour[i];
        if (!insideAllowedVerticalBand(p, boneMask.size())) continue;
        if (nearBorder(p, boneBox)) continue;
        int n = static_cast<int>(contour.size());
        int prev = (i - DIRECTION_OFFSET + n) % n;
        int next = (i + DIRECTION_OFFSET) % n;
        double dirChange = directionChangeDeg(contour[prev], p, contour[next]);
        cv::Rect roi = safeBox(p, boneMask.size(), EDGE_ROI_RADIUS);
        int edgeCount = cv::countNonZero(edges(roi));
        double edgeDensity = static_cast<double>(edgeCount) / static_cast<double>(roi.area());
        if (!validExternalFracture(dirChange, edgeCount, edgeDensity)) continue;
        Fracture f;
        f.regionId = externalId++;
        f.method = "EXTERNAL";
        f.location = p;
        f.score = dirChange + edgeCount + edgeDensity * 1000.0;
        f.box = shiftBoxUp(
            adaptiveFractureBox(p, boneMask.size(), boneMask, edges),
            EXTERNAL_BOX_SHIFT_UP,
            boneMask.size()
        );
        candidates.push_back(f);
    }
    std::sort(candidates.begin(), candidates.end(), [](const Fracture& a, const Fracture& b) {
        return a.score > b.score;
        });
    std::vector<Fracture> fractures;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const Fracture& c = candidates[i];
        bool tooClose = false;
        for (size_t j = 0; j < fractures.size(); ++j) {
            if (cv::norm(c.location - fractures[j].location) < EXTERNAL_MIN_DISTANCE_BETWEEN_FRACTURES) {
                tooClose = true;
                break;
            }
        }
        if (!tooClose) fractures.push_back(c);
        if (static_cast<int>(fractures.size()) >= MAX_FRACTURES_TO_SHOW) break;
    }
    return fractures;
}
cv::Mat createInnerBoneMask(const cv::Mat& boneMask, int marginFromContour) {
    cv::Mat innerMask;
    cv::erode(boneMask, innerMask, kernel(marginFromContour));
    return innerMask;
}
cv::Mat computeInternalGradientMagnitude(const cv::Mat& grayEnhanced, const cv::Mat& innerBoneMask) {
    cv::Mat blurred;
    cv::GaussianBlur(grayEnhanced, blurred, cv::Size(0, 0), INTERNAL_GRADIENT_BLUR_SIGMA);
    cv::Mat gradX;
    cv::Mat gradY;
    cv::Mat gradMag;
    cv::Sobel(blurred, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(blurred, gradY, CV_32F, 0, 1, 3);
    cv::magnitude(gradX, gradY, gradMag);
    gradMag *= SOBEL_SCALE_FACTOR;
    cv::Mat localMean;
    cv::GaussianBlur(grayEnhanced, localMean, cv::Size(0, 0), 3.0);
    cv::Mat grayFloat;
    cv::Mat localMeanFloat;
    grayEnhanced.convertTo(grayFloat, CV_32F);
    localMean.convertTo(localMeanFloat, CV_32F);
    cv::Mat localContrast;
    cv::absdiff(grayFloat, localMeanFloat, localContrast);
    std::vector<double> contrastValues;
    contrastValues.reserve(static_cast<size_t>(cv::countNonZero(innerBoneMask)));
    for (int y = 0; y < localContrast.rows; ++y) {
        const float* c = localContrast.ptr<float>(y);
        const uchar* m = innerBoneMask.ptr<uchar>(y);
        for (int x = 0; x < localContrast.cols; ++x) {
            if (m[x]) contrastValues.push_back(c[x]);
        }
    }
    double p90 = percentile(contrastValues, 0.90);
    cv::Mat localContrastNorm = cv::Mat::zeros(localContrast.size(), CV_32F);
    if (p90 > 1e-6) {
        localContrast.convertTo(localContrastNorm, CV_32F, 1.0 / p90);
        cv::min(localContrastNorm, 1.0, localContrastNorm);
    }
    cv::Mat amplification = cv::Mat::ones(localContrastNorm.size(), CV_32F);
    amplification += LOCAL_GRAY_CONTRAST_WEIGHT * localContrastNorm;
    cv::Mat sensitiveGradient;
    cv::multiply(gradMag, amplification, sensitiveGradient);
    sensitiveGradient.setTo(0, ~innerBoneMask);
    return sensitiveGradient;
}
InternalAdaptiveThresholds computeAdaptiveThresholdsFromDebug(const std::vector<InternalGradientDebug>& infos) {
    std::vector<double> meanGradValues;
    std::vector<double> stdGradValues;
    std::vector<double> maxGradValues;
    std::vector<double> stdGrayValues;
    std::vector<double> darkDiffValues;
    for (size_t i = 0; i < infos.size(); ++i) {
        meanGradValues.push_back(infos[i].meanGradient);
        stdGradValues.push_back(infos[i].stdGradient);
        maxGradValues.push_back(infos[i].maxGradient);
        stdGrayValues.push_back(infos[i].stdGray);
        darkDiffValues.push_back(infos[i].darkDifference);
    }
    InternalAdaptiveThresholds t;
    t.meanGray = INTERNAL_MEAN_GRAY_MIN_THRESHOLD;
    t.meanGradient = clampDouble(
        percentile(meanGradValues, ADAPTIVE_MEAN_GRADIENT_PERCENTILE),
        INTERNAL_GRADIENT_MEAN_MIN_THRESHOLD,
        INTERNAL_GRADIENT_MEAN_MAX_THRESHOLD
    );
    t.stdGradient = clampDouble(
        percentile(stdGradValues, ADAPTIVE_STD_GRADIENT_PERCENTILE),
        INTERNAL_GRADIENT_STD_MIN_THRESHOLD,
        INTERNAL_GRADIENT_STD_MAX_THRESHOLD
    );
    t.maxGradient = clampDouble(
        percentile(maxGradValues, ADAPTIVE_MAX_GRADIENT_PERCENTILE),
        INTERNAL_GRADIENT_PEAK_MIN_THRESHOLD,
        INTERNAL_GRADIENT_PEAK_MAX_THRESHOLD
    );
    t.stdGray = clampDouble(
        percentile(stdGrayValues, ADAPTIVE_GRAY_STD_PERCENTILE),
        INTERNAL_GRAY_STD_MIN_THRESHOLD,
        INTERNAL_GRAY_STD_MAX_THRESHOLD
    );
    t.darkDifference = clampDouble(
        percentile(darkDiffValues, ADAPTIVE_DARK_DIFF_PERCENTILE),
        INTERNAL_DARK_LINE_MIN_THRESHOLD,
        INTERNAL_DARK_LINE_MAX_THRESHOLD
    );
    return t;
}
InternalDecisionDebug evaluateInternalFractureDecision(
    const InternalGradientDebug& info,
    const InternalAdaptiveThresholds& t
) {
    InternalDecisionDebug d;
    d.passMeanGray = info.meanGray >= t.meanGray;
    d.passMeanGradient = info.meanGradient >= t.meanGradient;
    d.passStdGradient = info.stdGradient >= t.stdGradient;
    d.passMaxGradient = info.maxGradient >= t.maxGradient;
    d.passStdGray = info.stdGray >= t.stdGray;
    d.passDarkDifference = info.darkDifference >= t.darkDifference;
    d.strongTests = 0;
    if (d.passMeanGradient) d.strongTests++;
    if (d.passStdGradient) d.strongTests++;
    if (d.passMaxGradient) d.strongTests++;
    if (d.passStdGray) d.strongTests++;
    if (d.passDarkDifference) d.strongTests++;
    d.veryStrongGradient =
        d.passMeanGradient &&
        d.passStdGradient &&
        d.passMaxGradient &&
        (d.passStdGray || d.passDarkDifference);
    d.balancedStrong = d.strongTests >= INTERNAL_MIN_STRONG_TESTS;
    d.possibleFracture =
        d.passMeanGray &&
        (d.veryStrongGradient || d.balancedStrong);
    return d;
}
bool isPossibleInternalFracture(const InternalGradientDebug& info, const InternalAdaptiveThresholds& t) {
    return evaluateInternalFractureDecision(info, t).possibleFracture;
}
cv::Point2d weightedClusterCenter(const std::vector<Fracture>& items) {
    double weightedSumX = 0.0;
    double weightedSumY = 0.0;
    double weightSum = 0.0;
    for (size_t i = 0; i < items.size(); ++i) {
        double weight = std::max(0.0001, items[i].score);
        weightedSumX += items[i].location.x * weight;
        weightedSumY += items[i].location.y * weight;
        weightSum += weight;
    }
    if (weightSum <= 0.0) return cv::Point2d(0.0, 0.0);
    return cv::Point2d(weightedSumX / weightSum, weightedSumY / weightSum);
}
double localSupportScore(const Fracture& candidate, const std::vector<Fracture>& clusterItems) {
    if (clusterItems.empty()) return 0.0;
    double support = 0.0;
    for (size_t i = 0; i < clusterItems.size(); ++i) {
        double distance = cv::norm(candidate.location - clusterItems[i].location);
        if (distance > CLUSTER_RADIUS) continue;
        double spatialWeight = 1.0 - distance / std::max(1.0, static_cast<double>(CLUSTER_RADIUS));
        double scoreWeight = std::sqrt(std::max(0.0001, clusterItems[i].score));
        support += spatialWeight * scoreWeight;
    }
    return support;
}
Fracture chooseBestClusterRepresentative(const std::vector<Fracture>& clusterItems, const cv::Size& imageSize) {
    (void)imageSize;
    if (clusterItems.empty()) UCAS_THROW("Cluster vuoto");
    if (clusterItems.size() == 1) return clusterItems[0];
    cv::Point2d center = weightedClusterCenter(clusterItems);
    double maxScore = 0.0;
    double maxDistance = 1.0;
    double maxSupport = 0.0;
    std::vector<double> supports(clusterItems.size(), 0.0);
    for (size_t i = 0; i < clusterItems.size(); ++i) {
        const Fracture& item = clusterItems[i];
        maxScore = std::max(maxScore, item.score);
        double distance = cv::norm(cv::Point2d(item.location.x, item.location.y) - center);
        maxDistance = std::max(maxDistance, distance);
        supports[i] = localSupportScore(item, clusterItems);
        maxSupport = std::max(maxSupport, supports[i]);
    }
    double bestRank = -DBL_MAX;
    Fracture best = clusterItems[0];
    for (size_t i = 0; i < clusterItems.size(); ++i) {
        const Fracture& item = clusterItems[i];
        double normalizedScore = item.score / std::max(0.0001, maxScore);
        double distance = cv::norm(cv::Point2d(item.location.x, item.location.y) - center);
        double normalizedCentrality = 1.0 - distance / std::max(1.0, maxDistance);
        normalizedCentrality = clampDouble(normalizedCentrality, 0.0, 1.0);
        double normalizedSupport = supports[i] / std::max(0.0001, maxSupport);
        double rank =
            CLUSTER_SCORE_WEIGHT * normalizedScore +
            CLUSTER_CENTRALITY_WEIGHT * normalizedCentrality +
            CLUSTER_LOCAL_SUPPORT_WEIGHT * normalizedSupport;
        if (rank > bestRank) {
            bestRank = rank;
            best = item;
        }
    }
    return best;
}
std::vector<Fracture> mergeCloseFracturesByClusterRepresentative(
    const std::vector<Fracture>& candidates,
    const cv::Size& imageSize
) {
    if (candidates.empty()) return std::vector<Fracture>();
    std::vector<Fracture> sortedCandidates = candidates;
    std::sort(sortedCandidates.begin(), sortedCandidates.end(), [](const Fracture& a, const Fracture& b) {
        return a.score > b.score;
        });
    std::vector<FractureCluster> clusters;
    for (size_t i = 0; i < sortedCandidates.size(); ++i) {
        const Fracture& candidate = sortedCandidates[i];
        int bestClusterIndex = -1;
        double bestDistance = DBL_MAX;
        for (size_t j = 0; j < clusters.size(); ++j) {
            cv::Point2d center = weightedClusterCenter(clusters[j].items);
            double distance = cv::norm(cv::Point2d(candidate.location.x, candidate.location.y) - center);
            if (distance < CLUSTER_RADIUS && distance < bestDistance) {
                bestDistance = distance;
                bestClusterIndex = static_cast<int>(j);
            }
        }
        if (bestClusterIndex >= 0) {
            clusters[bestClusterIndex].items.push_back(candidate);
        }
        else {
            FractureCluster cluster;
            cluster.items.push_back(candidate);
            clusters.push_back(cluster);
        }
    }
    std::vector<ClusterRepresentative> representatives;
    for (size_t i = 0; i < clusters.size(); ++i) {
        if (clusters[i].items.empty()) continue;
        Fracture representative = chooseBestClusterRepresentative(clusters[i].items, imageSize);
        double clusterScore = 0.0;
        double maxScore = 0.0;
        for (size_t j = 0; j < clusters[i].items.size(); ++j) {
            clusterScore += std::sqrt(std::max(0.0001, clusters[i].items[j].score));
            maxScore = std::max(maxScore, clusters[i].items[j].score);
        }
        representative.score = maxScore + clusterScore * 0.15;
        ClusterRepresentative cr;
        cr.fracture = representative;
        cr.clusterScore = representative.score;
        cr.clusterSize = static_cast<int>(clusters[i].items.size());
        representatives.push_back(cr);
    }
    std::sort(representatives.begin(), representatives.end(), [](const ClusterRepresentative& a, const ClusterRepresentative& b) {
        if (std::abs(a.clusterScore - b.clusterScore) > 1e-6) {
            return a.clusterScore > b.clusterScore;
        }
        return a.clusterSize > b.clusterSize;
        });
    std::vector<Fracture> fractures;
    for (size_t i = 0; i < representatives.size(); ++i) {
        bool tooClose = false;
        for (size_t j = 0; j < fractures.size(); ++j) {
            if (cv::norm(representatives[i].fracture.location - fractures[j].location) < INTERNAL_MIN_DISTANCE_BETWEEN_FRACTURES) {
                tooClose = true;
                break;
            }
        }
        if (!tooClose) fractures.push_back(representatives[i].fracture);
        if (static_cast<int>(fractures.size()) >= MAX_FRACTURES_TO_SHOW) break;
    }
    return fractures;
}
std::vector<Fracture> detectInternalFracturesByGradientAndGray(
    const cv::Mat& grayEnhanced,
    const cv::Mat& boneMask,
    cv::Mat& innerBoneMask,
    cv::Mat& internalGradient,
    std::vector<InternalGradientDebug>& debugInfos
) {
    std::vector<Fracture> candidates;
    debugInfos.clear();
    innerBoneMask = createInnerBoneMask(boneMask, INTERNAL_MARGIN_FROM_CONTOUR);
    if (cv::countNonZero(innerBoneMask) == 0) return std::vector<Fracture>();
    internalGradient = computeInternalGradientMagnitude(grayEnhanced, innerBoneMask);
    cv::Rect innerBox = cv::boundingRect(innerBoneMask);
    int regionCounter = 1;
    for (int y = innerBox.y; y < innerBox.y + innerBox.height; y += INTERNAL_SCAN_STEP) {
        for (int x = innerBox.x; x < innerBox.x + innerBox.width; x += INTERNAL_SCAN_STEP) {
            cv::Point p(x, y);
            if (innerBoneMask.at<uchar>(p) == 0) continue;
            if (!insideAllowedVerticalBand(p, grayEnhanced.size())) continue;
            cv::Rect roi = safeBox(p, grayEnhanced.size(), INTERNAL_ROI_RADIUS);
            cv::Mat roiGray = grayEnhanced(roi);
            cv::Mat roiMask = innerBoneMask(roi);
            cv::Mat roiGradient = internalGradient(roi);
            int maskPixels = cv::countNonZero(roiMask);
            if (maskPixels < INTERNAL_MIN_MASK_PIXELS) continue;
            cv::Scalar meanGray;
            cv::Scalar stdGray;
            cv::meanStdDev(roiGray, meanGray, stdGray, roiMask);
            double minGray = 0.0;
            double maxGray = 0.0;
            cv::minMaxLoc(roiGray, &minGray, &maxGray, 0, 0, roiMask);
            cv::Scalar meanGrad;
            cv::Scalar stdGrad;
            cv::meanStdDev(roiGradient, meanGrad, stdGrad, roiMask);
            double minGrad = 0.0;
            double maxGrad = 0.0;
            cv::minMaxLoc(roiGradient, &minGrad, &maxGrad, 0, 0, roiMask);
            InternalGradientDebug info;
            info.id = regionCounter++;
            info.roi = roi;
            info.center = p;
            info.meanGray = meanGray[0];
            info.stdGray = stdGray[0];
            info.meanGradient = meanGrad[0];
            info.stdGradient = stdGrad[0];
            info.maxGradient = maxGrad;
            info.darkDifference = meanGray[0] - minGray;
            info.possibleFracture = false;
            debugInfos.push_back(info);
        }
    }
    if (debugInfos.empty()) return std::vector<Fracture>();
    InternalAdaptiveThresholds adaptiveThresholds = computeAdaptiveThresholdsFromDebug(debugInfos);
    for (size_t i = 0; i < debugInfos.size(); ++i) {
        InternalGradientDebug& info = debugInfos[i];
        info.possibleFracture = isPossibleInternalFracture(info, adaptiveThresholds);
        if (!info.possibleFracture) continue;
        Fracture f;
        f.regionId = info.id;
        f.method = "INTERNAL";
        f.location = info.center;
        f.score = info.meanGradient + info.stdGradient * 1.3 + info.maxGradient + info.stdGray * 2.0 + std::max(0.0, info.darkDifference) * 2.0;
        f.box = adaptiveFractureBox(info.center, grayEnhanced.size(), boneMask, internalGradient);
        candidates.push_back(f);
    }
    return mergeCloseFracturesByClusterRepresentative(candidates, grayEnhanced.size());
}
void drawSingleFracture(cv::Mat& image, const Fracture& f, const cv::Scalar& color, int thickness) {
    cv::rectangle(image, f.box, color, thickness);
    cv::circle(image, f.location, 4, color, -1);
    std::string label = f.method + " #" + std::to_string(f.regionId);
    cv::putText(image, label, cv::Point(f.box.x, std::max(15, f.box.y - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2);
}
void drawFractures(cv::Mat& image, const std::vector<Fracture>& fractures, const cv::Scalar& color, int thickness) {
    for (size_t i = 0; i < fractures.size(); ++i) {
        drawSingleFracture(image, fractures[i], color, thickness);
    }
}
void drawGroundTruthBoxes(cv::Mat& image, const std::vector<GroundTruthBox>& groundTruths, const cv::Scalar& color, int thickness) {
    for (size_t i = 0; i < groundTruths.size(); ++i) {
        cv::rectangle(image, groundTruths[i].box, color, thickness);
        std::string label = "GT frattura";
        cv::putText(
            image,
            label,
            cv::Point(groundTruths[i].box.x, std::max(15, groundTruths[i].box.y - 5)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        );
    }
}
std::vector<Fracture> mergeAllFractures(
    const std::vector<Fracture>& internalFractures,
    const std::vector<Fracture>& externalFractures
) {
    std::vector<Fracture> all;
    for (size_t i = 0; i < internalFractures.size(); ++i) all.push_back(internalFractures[i]);
    for (size_t i = 0; i < externalFractures.size(); ++i) all.push_back(externalFractures[i]);
    std::sort(all.begin(), all.end(), [](const Fracture& a, const Fracture& b) {
        return a.score > b.score;
        });
    if (static_cast<int>(all.size()) > MAX_FRACTURES_TO_SHOW) {
        all.resize(MAX_FRACTURES_TO_SHOW);
    }
    return all;
}
std::string getFileNameWithoutExtension(const std::string& path) {
    size_t slashPos = path.find_last_of("\\/");
    std::string filename = (slashPos == std::string::npos) ? path : path.substr(slashPos + 1);
    size_t dotPos = filename.find_last_of('.');
    if (dotPos == std::string::npos) return filename;
    return filename.substr(0, dotPos);
}
std::string buildLabelPathFromImagePath(const std::string& imagePath) {
    std::string baseName = getFileNameWithoutExtension(imagePath);
    return LABELS_DIR + "/" + baseName + ".txt";
}
cv::Rect clampRectToImage(const cv::Rect& r, const cv::Size& imageSize) {
    int x1 = clampInt(r.x, 0, imageSize.width - 1);
    int y1 = clampInt(r.y, 0, imageSize.height - 1);
    int x2 = clampInt(r.x + r.width, 0, imageSize.width);
    int y2 = clampInt(r.y + r.height, 0, imageSize.height);
    int w = std::max(1, x2 - x1);
    int h = std::max(1, y2 - y1);
    return cv::Rect(x1, y1, w, h);
}

cv::Rect expandFeatureExtractionRoi(const cv::Rect& roi, const cv::Size& imageSize) {
    cv::Rect baseRoi = clampRectToImage(roi, imageSize);

    int marginX = clampInt(
        static_cast<int>(std::round(static_cast<double>(baseRoi.width) * FEATURE_ROI_CONTEXT_RATIO)),
        FEATURE_ROI_CONTEXT_MIN_PIXELS,
        FEATURE_ROI_CONTEXT_MAX_PIXELS
    );
    int marginY = clampInt(
        static_cast<int>(std::round(static_cast<double>(baseRoi.height) * FEATURE_ROI_CONTEXT_RATIO)),
        FEATURE_ROI_CONTEXT_MIN_PIXELS,
        FEATURE_ROI_CONTEXT_MAX_PIXELS
    );

    cv::Rect expandedRoi(
        baseRoi.x - marginX,
        baseRoi.y - marginY,
        baseRoi.width + 2 * marginX,
        baseRoi.height + 2 * marginY
    );
    return clampRectToImage(expandedRoi, imageSize);
}
bool allBoxValuesLookNormalized(double x, double y, double w, double h) {
    return x >= 0.0 && x <= 1.0 &&
        y >= 0.0 && y <= 1.0 &&
        w > 0.0 && w <= 1.0 &&
        h > 0.0 && h <= 1.0;
}
cv::Rect yoloNormalizedToRect(double xCenter, double yCenter, double width, double height, const cv::Size& imageSize) {
    double boxWidth = width * static_cast<double>(imageSize.width);
    double boxHeight = height * static_cast<double>(imageSize.height);
    double x = xCenter * static_cast<double>(imageSize.width) - boxWidth / 2.0;
    double y = yCenter * static_cast<double>(imageSize.height) - boxHeight / 2.0;
    cv::Rect r(
        static_cast<int>(std::round(x)),
        static_cast<int>(std::round(y)),
        static_cast<int>(std::round(boxWidth)),
        static_cast<int>(std::round(boxHeight))
    );
    return clampRectToImage(r, imageSize);
}
cv::Rect pixelXywhToRect(double x, double y, double width, double height, const cv::Size& imageSize) {
    cv::Rect r(
        static_cast<int>(std::round(x)),
        static_cast<int>(std::round(y)),
        static_cast<int>(std::round(width)),
        static_cast<int>(std::round(height))
    );
    return clampRectToImage(r, imageSize);
}
std::vector<double> parseNumbersFromLine(const std::string& line) {
    std::vector<double> values;
    std::stringstream ss(line);
    double value = 0.0;
    while (ss >> value) {
        values.push_back(value);
    }
    return values;
}
std::vector<GroundTruthBox> loadGroundTruthBoxes(const std::string& imagePath, const cv::Size& imageSize, bool verboseOutput) {
    std::vector<GroundTruthBox> groundTruths;
    std::string labelPath = buildLabelPathFromImagePath(imagePath);
    std::ifstream file(labelPath.c_str());
    if (!file.is_open()) {
        if (verboseOutput) {
            std::cout << std::endl;
            std::cout << "ATTENZIONE: file etichetta non trovato: " << labelPath << std::endl;
            std::cout << "La valutazione IoU non verra' calcolata." << std::endl;
        }
        return groundTruths;
    }
    std::string line;
    int lineNumber = 0;
    int ignoredOtherClasses = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        std::vector<double> values = parseNumbersFromLine(line);
        if (values.size() != 5) {
            if (verboseOutput) {
                std::cout << "Riga etichetta ignorata: servono 5 valori nel formato class x_center y_center width height alla riga "
                    << lineNumber << ": " << line << std::endl;
            }
            continue;
        }
        int classId = static_cast<int>(std::round(values[0]));
        if (classId != FRACTURE_LABEL_CLASS_ID) {
            ++ignoredOtherClasses;
            continue;
        }
        double x = values[1];
        double y = values[2];
        double w = values[3];
        double h = values[4];
        if (w <= 0.0 || h <= 0.0) {
            if (verboseOutput) {
                std::cout << "Riga etichetta ignorata, dimensioni non valide alla riga "
                    << lineNumber << ": " << line << std::endl;
            }
            continue;
        }
        cv::Rect box;
        if (allBoxValuesLookNormalized(x, y, w, h)) {
            box = yoloNormalizedToRect(x, y, w, h, imageSize);
        }
        else {
            box = pixelXywhToRect(x, y, w, h, imageSize);
        }
        GroundTruthBox gt;
        gt.classId = classId;
        gt.box = box;
        gt.matched = false;
        groundTruths.push_back(gt);
    }
    if (verboseOutput) {
        std::cout << std::endl;
        std::cout << "File etichetta letto: " << labelPath << std::endl;
        std::cout << "Numero ground truth frattura considerate: " << groundTruths.size() << std::endl;
        std::cout << "Numero righe ignorate perche appartengono ad altre classi: " << ignoredOtherClasses << std::endl;
    }
    return groundTruths;
}
double computeIntersectionArea(const cv::Rect& a, const cv::Rect& b) {
    int xLeft = std::max(a.x, b.x);
    int yTop = std::max(a.y, b.y);
    int xRight = std::min(a.x + a.width, b.x + b.width);
    int yBottom = std::min(a.y + a.height, b.y + b.height);
    int intersectionWidth = std::max(0, xRight - xLeft);
    int intersectionHeight = std::max(0, yBottom - yTop);
    return static_cast<double>(intersectionWidth) * static_cast<double>(intersectionHeight);
}
double computeIoU(const cv::Rect& a, const cv::Rect& b) {
    double intersectionArea = computeIntersectionArea(a, b);
    double areaA = static_cast<double>(a.width) * static_cast<double>(a.height);
    double areaB = static_cast<double>(b.width) * static_cast<double>(b.height);
    double unionArea = areaA + areaB - intersectionArea;
    if (unionArea <= 0.0) return 0.0;
    return intersectionArea / unionArea;
}
EvaluationStats evaluateDetectionsWithIoU(
    const std::vector<Fracture>& detections,
    std::vector<GroundTruthBox>& groundTruths,
    double iouThreshold,
    bool verboseOutput
) {
    EvaluationStats stats;

    for (size_t i = 0; i < groundTruths.size(); ++i) {
        groundTruths[i].matched = false;
    }

    std::vector<Fracture> sortedDetections = detections;
    std::sort(sortedDetections.begin(), sortedDetections.end(), [](const Fracture& a, const Fracture& b) {
        return a.score > b.score;
        });

    for (size_t i = 0; i < sortedDetections.size(); ++i) {
        const Fracture& det = sortedDetections[i];

        double bestIoU = 0.0;
        double bestIntersectionArea = 0.0;
        int bestGtIndex = -1;

        for (size_t j = 0; j < groundTruths.size(); ++j) {
            if (groundTruths[j].matched) continue;

            double iou = computeIoU(det.box, groundTruths[j].box);
            double intersectionArea = computeIntersectionArea(det.box, groundTruths[j].box);

            if (iou > bestIoU || (std::abs(iou - bestIoU) < 1e-9 && intersectionArea > bestIntersectionArea)) {
                bestIoU = iou;
                bestIntersectionArea = intersectionArea;
                bestGtIndex = static_cast<int>(j);
            }
        }

        if (bestGtIndex >= 0 && bestIoU >= iouThreshold) {
            groundTruths[bestGtIndex].matched = true;
            stats.tp++;

            if (verboseOutput) {
                std::cout << "TP | " << det.method
                    << " ROI #" << det.regionId
                    << " | IoU = " << bestIoU
                    << std::endl;
            }
        }
        else {
            stats.fp++;

            if (verboseOutput) {
                std::cout << "FP | " << det.method
                    << " ROI #" << det.regionId
                    << " | migliore IoU = " << bestIoU
                    << std::endl;
            }
        }
    }

    for (size_t i = 0; i < groundTruths.size(); ++i) {
        if (!groundTruths[i].matched) {
            stats.fn++;
            if (verboseOutput) {
                std::cout << "FN | box GT non trovata = " << groundTruths[i].box << std::endl;
            }
        }
    }

    return stats;
}

void addImagesWithExtension(
    const std::string& imagesDir,
    const std::string& extension,
    std::vector<cv::String>& imagePaths
) {
    std::vector<cv::String> matches;
    cv::glob(imagesDir + "/*" + extension, matches, false);

    for (size_t i = 0; i < matches.size(); ++i) {
        imagePaths.push_back(matches[i]);
    }
}


std::string trimPathInput(const std::string& input) {
    std::string out = input;
    while (!out.empty() && (out[0] == ' ' || out[0] == '\t' || out[0] == '"')) {
        out.erase(out.begin());
    }
    while (!out.empty() && (out[out.size() - 1] == ' ' || out[out.size() - 1] == '\t' || out[out.size() - 1] == '"')) {
        out.erase(out.end() - 1);
    }
    return out;
}

std::string toLowerString(std::string value) {
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] >= 'A' && value[i] <= 'Z') {
            value[i] = static_cast<char>(value[i] - 'A' + 'a');
        }
    }
    return value;
}

bool hasSupportedImageExtension(const std::string& path) {
    std::string lowerPath = toLowerString(path);
    std::vector<std::string> extensions;
    extensions.push_back(".png");
    extensions.push_back(".jpg");
    extensions.push_back(".jpeg");
    extensions.push_back(".bmp");
    extensions.push_back(".tif");
    extensions.push_back(".tiff");

    for (size_t i = 0; i < extensions.size(); ++i) {
        const std::string& ext = extensions[i];
        if (lowerPath.size() >= ext.size() &&
            lowerPath.substr(lowerPath.size() - ext.size()) == ext) {
            return true;
        }
    }
    return false;
}

std::vector<cv::String> collectImagePaths(const std::string& imagesDir);

std::vector<cv::String> collectImagePathsFromFileOrFolder(const std::string& inputPath) {
    std::vector<cv::String> imagePaths;

    if (hasSupportedImageExtension(inputPath)) {
        imagePaths.push_back(inputPath);
        return imagePaths;
    }

    imagePaths = collectImagePaths(inputPath);
    return imagePaths;
}

std::vector<cv::String> collectImagePaths(const std::string& imagesDir) {
    std::vector<cv::String> imagePaths;

    addImagesWithExtension(imagesDir, ".png", imagePaths);
    addImagesWithExtension(imagesDir, ".jpg", imagePaths);
    addImagesWithExtension(imagesDir, ".jpeg", imagePaths);
    addImagesWithExtension(imagesDir, ".bmp", imagePaths);
    addImagesWithExtension(imagesDir, ".tif", imagePaths);
    addImagesWithExtension(imagesDir, ".tiff", imagePaths);

    addImagesWithExtension(imagesDir, ".PNG", imagePaths);
    addImagesWithExtension(imagesDir, ".JPG", imagePaths);
    addImagesWithExtension(imagesDir, ".JPEG", imagePaths);
    addImagesWithExtension(imagesDir, ".BMP", imagePaths);
    addImagesWithExtension(imagesDir, ".TIF", imagePaths);
    addImagesWithExtension(imagesDir, ".TIFF", imagePaths);

    std::sort(imagePaths.begin(), imagePaths.end());
    imagePaths.erase(std::unique(imagePaths.begin(), imagePaths.end()), imagePaths.end());

    return imagePaths;
}


struct TextureFeatures {
    // Feature usate realmente dal filtro ML: GLCM + LBP + HOG.
    std::vector<double> lbpHist;
    std::vector<double> hogHist;
    double glcmContrast;
    double glcmDissimilarity;
    double glcmHomogeneity;
    double glcmASM;
    double glcmEnergy;
    double glcmEntropy;
    double glcmCorrelation;

    TextureFeatures()
        : lbpHist(256, 0.0),
        hogHist(),
        glcmContrast(0.0),
        glcmDissimilarity(0.0),
        glcmHomogeneity(0.0),
        glcmASM(0.0),
        glcmEnergy(0.0),
        glcmEntropy(0.0),
        glcmCorrelation(0.0) {
    }
};

struct MedianRoiSize {
    cv::Size originalMedian;
    cv::Size hogSize;
    int roiCount;
    MedianRoiSize()
        : originalMedian(64, 128),
        hogSize(64, 128),
        roiCount(0) {
    }
};

std::string sanitizeFileName(const std::string& name) {
    std::string out = name;
    for (size_t i = 0; i < out.size(); ++i) {
        char c = out[i];
        bool ok = (c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '_' || c == '-';
        if (!ok) out[i] = '_';
    }
    return out;
}

std::string joinPath(const std::string& dir, const std::string& fileName) {
    if (dir.empty()) return fileName;
    char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\') return dir + fileName;
    return dir + "/" + fileName;
}

bool pathExists(const std::string& path) {
    struct stat info;
    return !path.empty() && stat(path.c_str(), &info) == 0;
}

void createDirectoryIfNeeded(const std::string& dir) {
    if (dir.empty()) return;
#ifdef _WIN32
    std::string command = "mkdir \"" + dir + "\" 2>nul";
#else
    std::string command = "mkdir -p \"" + dir + "\"";
#endif
    system(command.c_str());
}

void clearPreviousResults(const std::string& dir) {
    if (dir.empty()) return;

    // Protezione: evita cancellazioni accidentali fuori dalla cartella risultati del progetto.
    if (dir.find("Bone_Fracture_Detection") == std::string::npos ||
        dir.find("risultati_rilevamento_fratture") == std::string::npos) {
        std::cout << "[WARN] Pulizia annullata: percorso non sicuro: " << dir << std::endl;
        return;
    }

#ifdef _WIN32
    std::string command = "if exist \"" + dir + "\" rmdir /S /Q \"" + dir + "\"";
#else
    std::string command = "rm -rf \"" + dir + "\"";
#endif
    system(command.c_str());
}

void printProgressMessage(const std::string& message) {
    std::cout << message << std::endl;
    std::cout.flush();
}

void printProgressCounter(
    const std::string& phase,
    size_t current,
    size_t total
) {
    double percent = 0.0;
    if (total > 0) {
        percent = (static_cast<double>(current) * 100.0) / static_cast<double>(total);
    }

    std::cout << "[PROGRESSO] "
        << phase
        << " "
        << current
        << "/"
        << total
        << " ("
        << std::fixed
        << std::setprecision(1)
        << percent
        << "%)"
        << std::endl;
    std::cout.flush();
}

void printProgressStage(
    const std::string& imagePath,
    const std::string& stage
) {
    std::cout << "[STATO] "
        << getFileNameWithoutExtension(imagePath)
        << " -> "
        << stage
        << std::endl;
    std::cout.flush();
}


int roundUpToMultiple(int value, int multiple) {
    if (multiple <= 0) return value;
    if (value <= 0) return multiple;
    return ((value + multiple - 1) / multiple) * multiple;
}

int medianInt(std::vector<int> values, int fallbackValue) {
    if (values.empty()) return fallbackValue;
    std::sort(values.begin(), values.end());
    int n = static_cast<int>(values.size());
    if (n % 2 == 1) return values[n / 2];
    return static_cast<int>(std::round((static_cast<double>(values[n / 2 - 1]) + static_cast<double>(values[n / 2])) / 2.0));
}

cv::Size makeValidHogWindowSize(const cv::Size& inputSize) {
    // HOG usa cella 8x8 e blocco 16x16: la finestra deve essere almeno 16x16
    // e conviene renderla multipla di 8 per avere descrittori sempre confrontabili.
    int width = roundUpToMultiple(std::max(16, inputSize.width), 8);
    int height = roundUpToMultiple(std::max(16, inputSize.height), 8);
    return cv::Size(width, height);
}

std::vector<double> computeHOGFeatureVector(const cv::Mat& roiGrayInput, const cv::Size& hogWindowSize) {
    std::vector<double> out;
    if (roiGrayInput.empty()) return out;

    cv::Size winSize = makeValidHogWindowSize(hogWindowSize);

    cv::Mat roiGray;
    if (roiGrayInput.type() != CV_8U) {
        cv::normalize(roiGrayInput, roiGray, 0, 255, cv::NORM_MINMAX, CV_8U);
    }
    else {
        roiGray = roiGrayInput.clone();
    }

    cv::Mat resized;
    cv::resize(roiGray, resized, winSize, 0.0, 0.0, cv::INTER_LINEAR);

    cv::HOGDescriptor hog(
        winSize,
        cv::Size(16, 16),
        cv::Size(8, 8),
        cv::Size(8, 8),
        9
    );

    std::vector<float> descriptors;
    hog.compute(resized, descriptors, cv::Size(0, 0), cv::Size(0, 0));
    out.resize(descriptors.size(), 0.0);
    for (size_t i = 0; i < descriptors.size(); ++i) {
        out[i] = static_cast<double>(descriptors[i]);
    }
    return out;
}

cv::Mat resizeRoiToMedianSize(const cv::Mat& roiGray, const cv::Size& medianSize) {
    if (roiGray.empty()) return roiGray.clone();
    cv::Mat resized;
    cv::resize(roiGray, resized, makeValidHogWindowSize(medianSize), 0.0, 0.0, cv::INTER_LINEAR);
    return resized;
}

std::vector<double> computeLBPHistogram(const cv::Mat& roiGray) {
    std::vector<double> hist(256, 0.0);
    if (roiGray.empty() || roiGray.rows < 3 || roiGray.cols < 3) return hist;

    int count = 0;
    for (int y = 1; y < roiGray.rows - 1; ++y) {
        for (int x = 1; x < roiGray.cols - 1; ++x) {
            uchar center = roiGray.at<uchar>(y, x);
            unsigned char code = 0;
            code |= (roiGray.at<uchar>(y - 1, x - 1) >= center) << 7;
            code |= (roiGray.at<uchar>(y - 1, x) >= center) << 6;
            code |= (roiGray.at<uchar>(y - 1, x + 1) >= center) << 5;
            code |= (roiGray.at<uchar>(y, x + 1) >= center) << 4;
            code |= (roiGray.at<uchar>(y + 1, x + 1) >= center) << 3;
            code |= (roiGray.at<uchar>(y + 1, x) >= center) << 2;
            code |= (roiGray.at<uchar>(y + 1, x - 1) >= center) << 1;
            code |= (roiGray.at<uchar>(y, x - 1) >= center) << 0;
            hist[static_cast<int>(code)] += 1.0;
            count++;
        }
    }

    if (count > 0) {
        for (size_t i = 0; i < hist.size(); ++i) hist[i] /= static_cast<double>(count);
    }
    return hist;
}

cv::Mat quantizeGrayLevels(const cv::Mat& roiGray, int levels) {
    cv::Mat quantized = cv::Mat::zeros(roiGray.size(), CV_8U);
    if (roiGray.empty()) return quantized;

    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(roiGray, &minVal, &maxVal);
    if (maxVal <= minVal) return quantized;

    for (int y = 0; y < roiGray.rows; ++y) {
        const uchar* src = roiGray.ptr<uchar>(y);
        uchar* dst = quantized.ptr<uchar>(y);
        for (int x = 0; x < roiGray.cols; ++x) {
            double normalized = (static_cast<double>(src[x]) - minVal) / (maxVal - minVal);
            int q = static_cast<int>(std::floor(normalized * static_cast<double>(levels)));
            q = clampInt(q, 0, levels - 1);
            dst[x] = static_cast<uchar>(q);
        }
    }
    return quantized;
}

struct HaarSubbands {
    cv::Mat LL;
    cv::Mat LH; // horizontal
    cv::Mat HL; // vertical
    cv::Mat HH; // diagonal
};

TextureFeatures computeTextureFeaturesLBPGLCMHOG(
    const cv::Mat& roiGrayInput,
    const cv::Size& hogWindowSize,
    const cv::Mat& grayEnhanced,
    const Fracture& fracture
) {
    // grayEnhanced e fracture restano nella firma per compatibilita' con il resto del codice,
    // ma il CSV finale usa solo GLCM, LBP e HOG.
    (void)grayEnhanced;
    (void)fracture;

    TextureFeatures features;
    if (roiGrayInput.empty()) return features;

    cv::Mat roiGray;
    if (roiGrayInput.type() != CV_8U) {
        cv::normalize(roiGrayInput, roiGray, 0, 255, cv::NORM_MINMAX, CV_8U);
    }
    else {
        roiGray = roiGrayInput.clone();
    }

    features.lbpHist = computeLBPHistogram(roiGray);
    features.hogHist = computeHOGFeatureVector(roiGray, hogWindowSize);

    const int levels = 16;
    cv::Mat q = quantizeGrayLevels(roiGray, levels);
    std::vector<cv::Point> offsets;
    offsets.push_back(cv::Point(1, 0));
    offsets.push_back(cv::Point(1, -1));
    offsets.push_back(cv::Point(0, -1));
    offsets.push_back(cv::Point(-1, -1));

    double contrast = 0.0;
    double dissimilarity = 0.0;
    double homogeneity = 0.0;
    double asmValue = 0.0;
    double entropy = 0.0;
    double correlation = 0.0;
    int validOffsets = 0;

    for (size_t oi = 0; oi < offsets.size(); ++oi) {
        cv::Mat glcm = cv::Mat::zeros(levels, levels, CV_64F);
        int pairs = 0;
        cv::Point off = offsets[oi];

        for (int y = 0; y < q.rows; ++y) {
            int yy = y + off.y;
            if (yy < 0 || yy >= q.rows) continue;
            for (int x = 0; x < q.cols; ++x) {
                int xx = x + off.x;
                if (xx < 0 || xx >= q.cols) continue;

                int i = q.at<uchar>(y, x);
                int j = q.at<uchar>(yy, xx);
                glcm.at<double>(i, j) += 1.0;
                pairs++;
            }
        }

        if (pairs <= 0) continue;
        glcm /= static_cast<double>(pairs);
        validOffsets++;

        double meanI = 0.0, meanJ = 0.0;
        for (int i = 0; i < levels; ++i) {
            for (int j = 0; j < levels; ++j) {
                double p = glcm.at<double>(i, j);
                meanI += static_cast<double>(i) * p;
                meanJ += static_cast<double>(j) * p;
            }
        }

        double stdI = 0.0, stdJ = 0.0;
        for (int i = 0; i < levels; ++i) {
            for (int j = 0; j < levels; ++j) {
                double p = glcm.at<double>(i, j);
                stdI += (static_cast<double>(i) - meanI) * (static_cast<double>(i) - meanI) * p;
                stdJ += (static_cast<double>(j) - meanJ) * (static_cast<double>(j) - meanJ) * p;
            }
        }
        stdI = std::sqrt(stdI);
        stdJ = std::sqrt(stdJ);

        double localContrast = 0.0;
        double localDissimilarity = 0.0;
        double localHomogeneity = 0.0;
        double localASM = 0.0;
        double localEntropy = 0.0;
        double localCorrelation = 0.0;

        for (int i = 0; i < levels; ++i) {
            for (int j = 0; j < levels; ++j) {
                double p = glcm.at<double>(i, j);
                if (p <= 0.0) continue;

                double diff = static_cast<double>(i - j);
                localContrast += diff * diff * p;
                localDissimilarity += std::abs(diff) * p;
                localHomogeneity += p / (1.0 + std::abs(diff));
                localASM += p * p;
                localEntropy += -p * std::log(p);

                if (stdI > 1e-12 && stdJ > 1e-12) {
                    localCorrelation +=
                        ((static_cast<double>(i) - meanI) * (static_cast<double>(j) - meanJ) * p) /
                        (stdI * stdJ);
                }
            }
        }

        contrast += localContrast;
        dissimilarity += localDissimilarity;
        homogeneity += localHomogeneity;
        asmValue += localASM;
        entropy += localEntropy;
        correlation += localCorrelation;
    }

    if (validOffsets > 0) {
        double inv = 1.0 / static_cast<double>(validOffsets);
        features.glcmContrast = contrast * inv;
        features.glcmDissimilarity = dissimilarity * inv;
        features.glcmHomogeneity = homogeneity * inv;
        features.glcmASM = asmValue * inv;
        features.glcmEnergy = std::sqrt(features.glcmASM);
        features.glcmEntropy = entropy * inv;
        features.glcmCorrelation = correlation * inv;
    }

    return features;
}

struct RoiLabelInfo {
    double bestIoU;
    int bestGtIndex;
    int label;
    RoiLabelInfo()
        : bestIoU(0.0),
        bestGtIndex(-1),
        label(0) {
    }
};

std::vector<RoiLabelInfo> assignLabelsToRoisByIoU(
    const std::vector<Fracture>& detections,
    std::vector<GroundTruthBox>& groundTruths,
    double iouThreshold
) {
    std::vector<RoiLabelInfo> labels(detections.size());

    for (size_t i = 0; i < groundTruths.size(); ++i) {
        groundTruths[i].matched = false;
    }

    std::vector<int> order;
    order.reserve(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
        order.push_back(static_cast<int>(i));
    }

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return detections[a].score > detections[b].score;
        });

    for (size_t oi = 0; oi < order.size(); ++oi) {
        int detIndex = order[oi];
        const Fracture& det = detections[detIndex];

        double bestIoU = 0.0;
        double bestIntersectionArea = 0.0;
        int bestGtIndex = -1;

        for (size_t j = 0; j < groundTruths.size(); ++j) {
            double iou = computeIoU(det.box, groundTruths[j].box);
            double intersectionArea = computeIntersectionArea(det.box, groundTruths[j].box);

            if (iou > bestIoU ||
                (std::abs(iou - bestIoU) < 1e-9 && intersectionArea > bestIntersectionArea)) {
                bestIoU = iou;
                bestIntersectionArea = intersectionArea;
                bestGtIndex = static_cast<int>(j);
            }
        }

        labels[detIndex].bestIoU = bestIoU;
        labels[detIndex].bestGtIndex = bestGtIndex;

        if (bestGtIndex >= 0 && bestIoU >= iouThreshold && !groundTruths[bestGtIndex].matched) {
            labels[detIndex].label = 1;
            groundTruths[bestGtIndex].matched = true;
        }
        else {
            labels[detIndex].label = 0;
        }
    }

    return labels;
}

void writeRoiFeatureCsvHeader(std::ofstream& csv, int hogFeatureCount) {
    // CSV finale: metadati + label + feature usate dal filtro ML: GLCM, LBP e HOG.
    csv << "image,roi_id,method,x,y,width,height,score,roi_file,best_iou,label";
    csv << ",glcm_contrast,glcm_dissimilarity,glcm_homogeneity,glcm_asm,glcm_energy,glcm_entropy,glcm_correlation";
    for (int i = 0; i < 256; ++i) {
        csv << ",lbp_" << i;
    }
    for (int i = 0; i < hogFeatureCount; ++i) {
        csv << ",hog_" << i;
    }
    csv << std::endl;
}

void writeGlcmFeatureCsvHeader(std::ofstream& csv) {
    // CSV solo GLCM: utile per addestrare/normalizzare separatamente le feature GLCM in Python.
    csv << "image,roi_id,method,x,y,width,height,score,roi_file,best_iou,label";
    csv << ",glcm_contrast,glcm_dissimilarity,glcm_homogeneity,glcm_asm,glcm_energy,glcm_entropy,glcm_correlation";
    csv << std::endl;
}

void writeLbpFeatureCsvHeader(std::ofstream& csv) {
    // CSV solo LBP: utile per addestrare/normalizzare separatamente l'istogramma LBP in Python.
    csv << "image,roi_id,method,x,y,width,height,score,roi_file,best_iou,label";
    for (int i = 0; i < 256; ++i) {
        csv << ",lbp_" << i;
    }
    csv << std::endl;
}

void writeHogFeatureCsvHeader(std::ofstream& csv, int hogFeatureCount) {
    csv << "image,roi_id,method,x,y,width,height,score,roi_file,best_iou,label";
    for (int i = 0; i < hogFeatureCount; ++i) {
        csv << ",hog_" << i;
    }
    csv << std::endl;
}

void writeFnCsvHeader(std::ofstream& csv) {
    csv << "image,fn_id,x,y,width,height,fn_file" << std::endl;
}

void writeCommonRoiMetadata(
    std::ofstream& csv,
    const std::string& imageName,
    int roiIndex,
    const Fracture& fracture,
    const std::string& roiFileName,
    double bestIoU,
    int label
) {
    csv << imageName << ",";
    csv << roiIndex << ",";
    csv << fracture.method << ",";
    csv << fracture.box.x << ",";
    csv << fracture.box.y << ",";
    csv << fracture.box.width << ",";
    csv << fracture.box.height << ",";
    csv << std::fixed << std::setprecision(6) << fracture.score << ",";
    csv << roiFileName << ",";
    csv << std::fixed << std::setprecision(6) << bestIoU << ",";
    csv << label;
}

void writeGlcmValues(std::ofstream& csv, const TextureFeatures& features) {
    csv << "," << std::fixed << std::setprecision(10) << features.glcmContrast;
    csv << "," << std::fixed << std::setprecision(10) << features.glcmDissimilarity;
    csv << "," << std::fixed << std::setprecision(10) << features.glcmHomogeneity;
    csv << "," << std::fixed << std::setprecision(10) << features.glcmASM;
    csv << "," << std::fixed << std::setprecision(10) << features.glcmEnergy;
    csv << "," << std::fixed << std::setprecision(10) << features.glcmEntropy;
    csv << "," << std::fixed << std::setprecision(10) << features.glcmCorrelation;
}

void writeLbpValues(std::ofstream& csv, const TextureFeatures& features) {
    for (size_t i = 0; i < features.lbpHist.size(); ++i) {
        csv << "," << std::fixed << std::setprecision(10) << features.lbpHist[i];
    }
}

void writeHogValues(std::ofstream& csv, const TextureFeatures& features, int hogFeatureCount) {
    for (int i = 0; i < hogFeatureCount; ++i) {
        double value = 0.0;
        if (i < static_cast<int>(features.hogHist.size())) value = features.hogHist[i];
        csv << "," << std::fixed << std::setprecision(10) << value;
    }
}
void writeRoiFeatureCsvRow(
    std::ofstream& csv,
    const std::string& imageName,
    int roiIndex,
    const Fracture& fracture,
    const std::string& roiFileName,
    double bestIoU,
    int label,
    const TextureFeatures& features,
    int hogFeatureCount
) {
    writeCommonRoiMetadata(csv, imageName, roiIndex, fracture, roiFileName, bestIoU, label);
    writeGlcmValues(csv, features);
    writeLbpValues(csv, features);
    writeHogValues(csv, features, hogFeatureCount);
    csv << std::endl;
}

void writeGlcmFeatureCsvRow(
    std::ofstream& csv,
    const std::string& imageName,
    int roiIndex,
    const Fracture& fracture,
    const std::string& roiFileName,
    double bestIoU,
    int label,
    const TextureFeatures& features
) {
    writeCommonRoiMetadata(csv, imageName, roiIndex, fracture, roiFileName, bestIoU, label);
    writeGlcmValues(csv, features);
    csv << std::endl;
}

void writeLbpFeatureCsvRow(
    std::ofstream& csv,
    const std::string& imageName,
    int roiIndex,
    const Fracture& fracture,
    const std::string& roiFileName,
    double bestIoU,
    int label,
    const TextureFeatures& features
) {
    writeCommonRoiMetadata(csv, imageName, roiIndex, fracture, roiFileName, bestIoU, label);
    writeLbpValues(csv, features);
    csv << std::endl;
}

void writeHogFeatureCsvRow(
    std::ofstream& csv,
    const std::string& imageName,
    int roiIndex,
    const Fracture& fracture,
    const std::string& roiFileName,
    double bestIoU,
    int label,
    const TextureFeatures& features,
    int hogFeatureCount
) {
    writeCommonRoiMetadata(csv, imageName, roiIndex, fracture, roiFileName, bestIoU, label);
    writeHogValues(csv, features, hogFeatureCount);
    csv << std::endl;
}

MedianRoiSize computeMedianRoiSizeFromImages(const std::vector<cv::String>& imagePaths) {
    MedianRoiSize result;
    std::vector<int> widths;
    std::vector<int> heights;

    printProgressMessage("");
    printProgressMessage("========================================");
    printProgressMessage("SCANSIONE PRELIMINARE ROI PER DIMENSIONE MEDIANA");
    printProgressMessage("========================================");
    printProgressMessage("Questa fase puo' richiedere tempo: per ogni immagine vengono rifatte segmentazione e ricerca ROI.");
    printProgressCounter("Scansione preliminare", 0, imagePaths.size());

    for (size_t i = 0; i < imagePaths.size(); ++i) {
        std::string imagePath = std::string(imagePaths[i]);
        printProgressCounter("Scansione preliminare", i + 1, imagePaths.size());
        printProgressStage(imagePath, "lettura immagine");

        try {
            int bitsUsed = 0;
            cv::Mat img = ucas::imread(imagePath, cv::IMREAD_UNCHANGED, &bitsUsed);
            if (img.empty()) {
                printProgressStage(imagePath, "immagine vuota, salto");
                continue;
            }

            printProgressStage(imagePath, "conversione/preparazione grigio");
            cv::Mat gray = improveImage(img, bitsUsed);

            printProgressStage(imagePath, "segmentazione osso + edges");
            cv::Mat boneMask;
            cv::Mat edges;
            computeSegmentation(gray, boneMask, edges);

            printProgressStage(imagePath, "enhancement immagine");
            cv::Mat grayEnhanced = enhanceGrayForDisplay(gray, boneMask);

            printProgressStage(imagePath, "ricerca fratture esterne");
            std::vector<Fracture> externalFractures = detectExternalFractures(boneMask, edges);

            printProgressStage(imagePath, "ricerca fratture interne");
            cv::Mat innerBoneMask;
            cv::Mat internalGradient;
            std::vector<InternalGradientDebug> debugInfos;
            std::vector<Fracture> internalFractures = detectInternalFracturesByGradientAndGray(
                grayEnhanced,
                boneMask,
                innerBoneMask,
                internalGradient,
                debugInfos
            );

            printProgressStage(imagePath, "merge ROI interne/esterne");
            std::vector<Fracture> allFractures = mergeAllFractures(internalFractures, externalFractures);

            int validRoisForImage = 0;
            for (size_t j = 0; j < allFractures.size(); ++j) {
                // La mediana HOG deve riflettere la ROI realmente usata per le feature,
                // cioe' il box originale con un piccolo contesto esterno.
                cv::Rect box = expandFeatureExtractionRoi(allFractures[j].box, grayEnhanced.size());
                if (box.width > 0 && box.height > 0) {
                    widths.push_back(box.width);
                    heights.push_back(box.height);
                    validRoisForImage++;
                }
            }

            std::cout << "[OK] "
                << getFileNameWithoutExtension(imagePath)
                << " -> ROI valide in scansione preliminare: "
                << validRoisForImage
                << " | ROI totali accumulate: "
                << widths.size()
                << std::endl;
            std::cout.flush();
        }
        catch (const std::exception& e) {
            std::cout << "[ERRORE] Scansione preliminare saltata per: "
                << imagePath
                << " | "
                << e.what()
                << std::endl;
            std::cout.flush();
        }
        catch (...) {
            std::cout << "[ERRORE] Scansione preliminare saltata per: "
                << imagePath
                << " | errore sconosciuto"
                << std::endl;
            std::cout.flush();
        }
    }

    result.roiCount = static_cast<int>(widths.size());
    result.originalMedian = cv::Size(
        medianInt(widths, 64),
        medianInt(heights, 128)
    );
    result.hogSize = makeValidHogWindowSize(result.originalMedian);

    std::cout << "[FINE] Scansione preliminare completata. ROI totali trovate: "
        << result.roiCount
        << " | Mediana: "
        << result.originalMedian.width
        << "x"
        << result.originalMedian.height
        << " | HOG window: "
        << result.hogSize.width
        << "x"
        << result.hogSize.height
        << std::endl;
    std::cout.flush();

    return result;
}

int extractRoisAndTextureFeaturesFromImage(
    const std::string& imagePath,
    const std::string& outputDir,
    std::ofstream& combinedCsv,
    std::ofstream& glcmCsv,
    std::ofstream& lbpCsv,
    std::ofstream& hogCsv,
    std::ofstream& fnCsv,
    const cv::Size& hogWindowSize,
    int hogFeatureCount
) {
    printProgressMessage("");
    printProgressMessage("----------------------------------------");
    printProgressMessage("INIZIO ESTRAZIONE ROI/CSV: " + imagePath);
    printProgressMessage("----------------------------------------");

    int bitsUsed = 0;

    printProgressStage(imagePath, "lettura immagine");
    cv::Mat img = ucas::imread(imagePath, cv::IMREAD_UNCHANGED, &bitsUsed);
    if (img.empty()) {
        UCAS_THROW(ucas::strprintf("Impossibile aprire l'immagine: %s", imagePath.c_str()));
    }

    printProgressStage(imagePath, "conversione/preparazione grigio");
    cv::Mat gray = improveImage(img, bitsUsed);

    printProgressStage(imagePath, "segmentazione osso + edges");
    cv::Mat boneMask;
    cv::Mat edges;
    computeSegmentation(gray, boneMask, edges);

    printProgressStage(imagePath, "enhancement immagine");
    cv::Mat grayEnhanced = enhanceGrayForDisplay(gray, boneMask);

    printProgressStage(imagePath, "ricerca fratture esterne");
    std::vector<Fracture> externalFractures = detectExternalFractures(boneMask, edges);
    std::cout << "[INFO] "
        << getFileNameWithoutExtension(imagePath)
        << " -> fratture esterne candidate: "
        << externalFractures.size()
        << std::endl;
    std::cout.flush();

    printProgressStage(imagePath, "ricerca fratture interne");
    cv::Mat innerBoneMask;
    cv::Mat internalGradient;
    std::vector<InternalGradientDebug> debugInfos;
    std::vector<Fracture> internalFractures = detectInternalFracturesByGradientAndGray(
        grayEnhanced,
        boneMask,
        innerBoneMask,
        internalGradient,
        debugInfos
    );
    std::cout << "[INFO] "
        << getFileNameWithoutExtension(imagePath)
        << " -> finestre interne analizzate: "
        << debugInfos.size()
        << " | fratture interne candidate dopo clustering: "
        << internalFractures.size()
        << std::endl;
    std::cout.flush();

    printProgressStage(imagePath, "merge ROI interne/esterne");
    std::vector<Fracture> allFractures = mergeAllFractures(internalFractures, externalFractures);
    std::cout << "[INFO] "
        << getFileNameWithoutExtension(imagePath)
        << " -> ROI totali da esportare: "
        << allFractures.size()
        << std::endl;
    std::cout.flush();

    std::string baseName = sanitizeFileName(getFileNameWithoutExtension(imagePath));

    printProgressStage(imagePath, "lettura label e assegnazione TP/FP");
    std::vector<GroundTruthBox> groundTruths = loadGroundTruthBoxes(imagePath, grayEnhanced.size(), false);
    std::vector<RoiLabelInfo> roiLabels = assignLabelsToRoisByIoU(
        allFractures,
        groundTruths,
        IOU_THRESHOLD
    );

    std::string tpDir = joinPath(outputDir, "TP");
    std::string fpDir = joinPath(outputDir, "FP");
    std::string fnDir = joinPath(outputDir, "FN");
    createDirectoryIfNeeded(tpDir);
    createDirectoryIfNeeded(fpDir);
    createDirectoryIfNeeded(fnDir);

    int savedRois = 0;
    int tpCount = 0;
    int fpCount = 0;

    printProgressCounter("ROI immagine " + baseName, 0, allFractures.size());

    for (size_t i = 0; i < allFractures.size(); ++i) {
        std::cout << "[ROI] "
            << baseName
            << " ROI "
            << (i + 1)
            << "/"
            << allFractures.size()
            << " -> preparo crop"
            << std::endl;
        std::cout.flush();

        Fracture fracture = allFractures[i];
        fracture.box = clampRectToImage(fracture.box, grayEnhanced.size());

        // Il box della detection rimane invariato per label, IoU e metadati.
        // Solo il crop da cui si estraggono le feature include poco contesto esterno.
        cv::Rect featureExtractionBox = expandFeatureExtractionRoi(fracture.box, grayEnhanced.size());
        cv::Mat roi = grayEnhanced(featureExtractionBox).clone();
        if (roi.empty()) {
            std::cout << "[ROI] "
                << baseName
                << " ROI "
                << (i + 1)
                << " vuota, salto"
                << std::endl;
            std::cout.flush();
            continue;
        }

        std::cout << "[ROI] "
            << baseName
            << " ROI "
            << (i + 1)
            << " -> resize/HOG window "
            << hogWindowSize.width
            << "x"
            << hogWindowSize.height
            << std::endl;
        std::cout.flush();

        cv::Mat roiForFeatures = resizeRoiToMedianSize(roi, hogWindowSize);

        std::cout << "[ROI] "
            << baseName
            << " ROI "
            << (i + 1)
            << " -> calcolo feature: GLCM, LBP, HOG"
            << std::endl;
        std::cout.flush();

        TextureFeatures features = computeTextureFeaturesLBPGLCMHOG(roiForFeatures, hogWindowSize, grayEnhanced, fracture);

        std::cout << "[ROI] "
            << baseName
            << " ROI "
            << (i + 1)
            << " -> feature calcolate, scrittura immagine e CSV"
            << std::endl;
        std::cout.flush();

        int label = 0;
        double bestIoU = 0.0;
        if (i < roiLabels.size()) {
            label = roiLabels[i].label;
            bestIoU = roiLabels[i].bestIoU;
        }

        std::string classFolder = (label == 1) ? "TP" : "FP";
        std::string classDir = (label == 1) ? tpDir : fpDir;
        if (label == 1) tpCount++;
        else fpCount++;

        std::stringstream roiNameStream;
        roiNameStream << baseName
            << "_roi_" << std::setw(3) << std::setfill('0') << (i + 1)
            << "_" << fracture.method
            << "_" << classFolder
            << ".png";
        std::string roiFileName = roiNameStream.str();
        std::string roiPath = joinPath(classDir, roiFileName);
        cv::imwrite(roiPath, roiForFeatures);

        std::string relativeRoiFile = joinPath(classFolder, roiFileName);

        writeRoiFeatureCsvRow(
            combinedCsv,
            baseName,
            static_cast<int>(i + 1),
            fracture,
            relativeRoiFile,
            bestIoU,
            label,
            features,
            hogFeatureCount
        );

        writeGlcmFeatureCsvRow(
            glcmCsv,
            baseName,
            static_cast<int>(i + 1),
            fracture,
            relativeRoiFile,
            bestIoU,
            label,
            features
        );

        writeLbpFeatureCsvRow(
            lbpCsv,
            baseName,
            static_cast<int>(i + 1),
            fracture,
            relativeRoiFile,
            bestIoU,
            label,
            features
        );

        writeHogFeatureCsvRow(
            hogCsv,
            baseName,
            static_cast<int>(i + 1),
            fracture,
            relativeRoiFile,
            bestIoU,
            label,
            features,
            hogFeatureCount
        );

        combinedCsv.flush();
        glcmCsv.flush();
        lbpCsv.flush();
        hogCsv.flush();

        savedRois++;

        std::cout << "[ROI OK] "
            << baseName
            << " ROI "
            << (i + 1)
            << "/"
            << allFractures.size()
            << " salvata come "
            << relativeRoiFile
            << " | label="
            << label
            << " | IoU="
            << std::fixed
            << std::setprecision(4)
            << bestIoU
            << std::endl;
        std::cout.flush();

        printProgressCounter("ROI immagine " + baseName, i + 1, allFractures.size());
    }

    printProgressStage(imagePath, "salvataggio FN mancate");
    int fnCount = 0;
    for (size_t i = 0; i < groundTruths.size(); ++i) {
        if (groundTruths[i].matched) continue;

        cv::Rect fnBox = clampRectToImage(groundTruths[i].box, grayEnhanced.size());
        cv::Mat fnCrop = grayEnhanced(fnBox).clone();

        std::stringstream fnNameStream;
        fnNameStream << baseName
            << "_fn_" << std::setw(3) << std::setfill('0') << (fnCount + 1)
            << ".png";
        std::string fnFileName = fnNameStream.str();
        std::string fnRelativeFile = joinPath("FN", fnFileName);
        std::string fnPath = joinPath(fnDir, fnFileName);

        if (!fnCrop.empty()) {
            cv::imwrite(fnPath, fnCrop);
        }

        fnCsv << baseName << ","
            << (fnCount + 1) << ","
            << fnBox.x << ","
            << fnBox.y << ","
            << fnBox.width << ","
            << fnBox.height << ","
            << fnRelativeFile
            << std::endl;
        fnCsv.flush();

        fnCount++;

        std::cout << "[FN] "
            << baseName
            << " FN salvata "
            << fnCount
            << "/"
            << groundTruths.size()
            << " -> "
            << fnRelativeFile
            << std::endl;
        std::cout.flush();
    }

    std::cout << "[FINE IMMAGINE] " << imagePath
        << " | ROI esportate: " << savedRois
        << " | TP: " << tpCount
        << " | FP: " << fpCount
        << " | FN: " << fnCount
        << std::endl;
    std::cout.flush();

    return savedRois;
}

void runRoiFeatureExtractionMode() {
    std::string inputPath;
    std::cout << "Inserisci il percorso dell'immagine oppure della cartella immagini"
        << " [default: " << DEFAULT_IMAGES_DIR << "]: ";
    std::getline(std::cin, inputPath);
    inputPath = trimPathInput(inputPath);
    if (inputPath.empty()) inputPath = DEFAULT_IMAGES_DIR;

    LABELS_DIR = DEFAULT_LABELS_DIR;
    std::cout << "Cartella immagini: " << inputPath << std::endl;
    std::cout << "Cartella label: " << LABELS_DIR << std::endl;

    if (!pathExists(LABELS_DIR)) {
        std::cout << "ATTENZIONE: cartella label non trovata:" << std::endl;
        std::cout << LABELS_DIR << std::endl;
        std::cout << "Controlla che le label siano in img_fracture/labels." << std::endl;
        return;
    }

    std::string outputDir = DEFAULT_RESULTS_DIR;
    std::string csvDir = joinPath(outputDir, "CSV_feature");
    std::string roiDir = joinPath(outputDir, "ROI_classificate");

    std::cout << "[STATO] Pulizia dei risultati precedenti..." << std::endl;
    clearPreviousResults(outputDir);

    createDirectoryIfNeeded(outputDir);
    createDirectoryIfNeeded(csvDir);
    createDirectoryIfNeeded(roiDir);

    std::cout << "Cartella risultati: " << outputDir << std::endl;
    std::cout << "Cartella CSV: " << csvDir << std::endl;
    std::cout << "Cartella ROI: " << roiDir << std::endl;

    std::cout << "[STATO] Raccolta immagini da: " << inputPath << std::endl;
    std::cout.flush();

    std::vector<cv::String> imagePaths = collectImagePathsFromFileOrFolder(inputPath);
    if (imagePaths.empty()) {
        std::cout << "Nessuna immagine trovata o percorso non valido:" << std::endl;
        std::cout << inputPath << std::endl;
        return;
    }

    std::cout << "[INFO] Immagini trovate per estrazione ROI/CSV: " << imagePaths.size() << std::endl;
    std::cout << "[STATO] Avvio scansione preliminare per calcolare dimensione mediana ROI..." << std::endl;
    std::cout.flush();

    MedianRoiSize medianInfo = computeMedianRoiSizeFromImages(imagePaths);
    // Finestra HOG fissa: mantiene costante il numero di colonne nel CSV.
    medianInfo.hogSize = cv::Size(96, 96);
    std::vector<double> emptyHogVector = computeHOGFeatureVector(
        cv::Mat::zeros(medianInfo.hogSize, CV_8U),
        medianInfo.hogSize
    );
    int hogFeatureCount = static_cast<int>(emptyHogVector.size());

    std::cout << "ROI trovate nella scansione preliminare: " << medianInfo.roiCount << std::endl;
    std::cout << "Dimensione mediana ROI originale: "
        << medianInfo.originalMedian.width << "x" << medianInfo.originalMedian.height << std::endl;
    std::cout << "Dimensione usata per resize/HOG: "
        << medianInfo.hogSize.width << "x" << medianInfo.hogSize.height << std::endl;
    std::cout << "Numero feature HOG: " << hogFeatureCount << std::endl;

    std::cout << "[STATO] Creazione CSV di output..." << std::endl;
    std::cout.flush();

    std::string csvPath = joinPath(csvDir, "roi_feature_glcm_lbp_hog_labeled.csv");
    std::ofstream combinedCsv(csvPath.c_str());
    if (!combinedCsv.is_open()) {
        UCAS_THROW(ucas::strprintf("Impossibile creare il file CSV completo: %s", csvPath.c_str()));
    }
    writeRoiFeatureCsvHeader(combinedCsv, hogFeatureCount);

    std::string glcmCsvPath = joinPath(csvDir, "roi_feature_glcm_labeled.csv");
    std::ofstream glcmCsv(glcmCsvPath.c_str());
    if (!glcmCsv.is_open()) {
        UCAS_THROW(ucas::strprintf("Impossibile creare il file CSV GLCM: %s", glcmCsvPath.c_str()));
    }
    writeGlcmFeatureCsvHeader(glcmCsv);

    std::string lbpCsvPath = joinPath(csvDir, "roi_feature_lbp_labeled.csv");
    std::ofstream lbpCsv(lbpCsvPath.c_str());
    if (!lbpCsv.is_open()) {
        UCAS_THROW(ucas::strprintf("Impossibile creare il file CSV LBP: %s", lbpCsvPath.c_str()));
    }
    writeLbpFeatureCsvHeader(lbpCsv);

    std::string hogCsvPath = joinPath(csvDir, "roi_feature_hog_labeled.csv");
    std::ofstream hogCsv(hogCsvPath.c_str());
    if (!hogCsv.is_open()) {
        UCAS_THROW(ucas::strprintf("Impossibile creare il file CSV HOG: %s", hogCsvPath.c_str()));
    }
    writeHogFeatureCsvHeader(hogCsv, hogFeatureCount);

    std::string fnCsvPath = joinPath(csvDir, "fratture_mancate_FN.csv");
    std::ofstream fnCsv(fnCsvPath.c_str());
    if (!fnCsv.is_open()) {
        UCAS_THROW(ucas::strprintf("Impossibile creare il file CSV FN: %s", fnCsvPath.c_str()));
    }
    writeFnCsvHeader(fnCsv);

    int processedImages = 0;
    int skippedImages = 0;
    int totalRois = 0;

    printProgressMessage("");
    printProgressMessage("========================================");
    printProgressMessage("AVVIO ESTRAZIONE ROI, FEATURE E CSV");
    printProgressMessage("========================================");
    printProgressCounter("Immagini elaborate", 0, imagePaths.size());

    for (size_t i = 0; i < imagePaths.size(); ++i) {
        try {
            std::cout << std::endl;
            std::cout << "[IMMAGINE] "
                << (i + 1)
                << "/"
                << imagePaths.size()
                << " -> "
                << imagePaths[i]
                << std::endl;
            std::cout.flush();

            totalRois += extractRoisAndTextureFeaturesFromImage(
                std::string(imagePaths[i]),
                roiDir,
                combinedCsv,
                glcmCsv,
                lbpCsv,
                hogCsv,
                fnCsv,
                medianInfo.hogSize,
                hogFeatureCount
            );
            processedImages++;

            combinedCsv.flush();
            glcmCsv.flush();
            lbpCsv.flush();
            hogCsv.flush();
            fnCsv.flush();

            printProgressCounter("Immagini elaborate", i + 1, imagePaths.size());
            std::cout << "[PARZIALE] Immagini elaborate: "
                << processedImages
                << " | saltate: "
                << skippedImages
                << " | ROI totali: "
                << totalRois
                << std::endl;
            std::cout.flush();
        }
        catch (const std::exception& e) {
            std::cout << "Errore durante estrazione feature da: " << imagePaths[i] << std::endl;
            std::cout << "Messaggio errore: " << e.what() << std::endl;
            skippedImages++;
        }
        catch (...) {
            std::cout << "Errore sconosciuto durante estrazione feature da: " << imagePaths[i] << std::endl;
            skippedImages++;
        }
    }

    combinedCsv.close();
    glcmCsv.close();
    lbpCsv.close();
    hogCsv.close();
    fnCsv.close();

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "ESTRAZIONE ROI E FEATURE COMPLETATA" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Immagini elaborate: " << processedImages << std::endl;
    std::cout << "Immagini saltate: " << skippedImages << std::endl;
    std::cout << "ROI totali esportate: " << totalRois << std::endl;
    std::cout << "Cartella output principale: " << outputDir << std::endl;
    std::cout << "Cartella CSV: " << csvDir << std::endl;
    std::cout << "Cartella ROI: " << roiDir << std::endl;
    std::cout << "File CSV completo con label: " << csvPath << std::endl;
    std::cout << "File CSV solo GLCM con label: " << glcmCsvPath << std::endl;
    std::cout << "File CSV solo LBP con label: " << lbpCsvPath << std::endl;
    std::cout << "File CSV solo HOG con label: " << hogCsvPath << std::endl;
    std::cout << "File CSV fratture mancate: " << fnCsvPath << std::endl;
    std::cout << "ROI vere fratture salvate in: " << joinPath(roiDir, "TP") << std::endl;
    std::cout << "ROI falsi positivi salvate in: " << joinPath(roiDir, "FP") << std::endl;
    std::cout << "Fratture mancate salvate in: " << joinPath(roiDir, "FN") << std::endl;
}

EvaluationStats processSingleImage(const std::string& imagePath, bool showImages, bool verboseOutput) {
    EvaluationStats stats;

    if (verboseOutput) {
        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Elaborazione immagine: " << imagePath << std::endl;
        std::cout << "========================================" << std::endl;
    }
    else {
        std::cout << std::endl;
        std::cout << "Elaborazione immagine: " << getFileNameWithoutExtension(imagePath) << std::endl;
    }

    int bitsUsed = 0;
    cv::Mat img = ucas::imread(imagePath, cv::IMREAD_UNCHANGED, &bitsUsed);
    if (img.empty()) {
        std::cout << "Impossibile aprire l'immagine: " << imagePath << std::endl;
        return stats;
    }

    cv::Mat gray = improveImage(img, bitsUsed);

    cv::Mat boneMask;
    cv::Mat edges;
    computeSegmentation(gray, boneMask, edges);

    cv::Mat grayEnhanced = enhanceGrayForDisplay(gray, boneMask);

    std::vector<Fracture> externalFractures = detectExternalFractures(boneMask, edges);

    cv::Mat innerBoneMask;
    cv::Mat internalGradient;
    std::vector<InternalGradientDebug> debugInfos;
    std::vector<Fracture> internalFractures = detectInternalFracturesByGradientAndGray(
        grayEnhanced,
        boneMask,
        innerBoneMask,
        internalGradient,
        debugInfos
    );

    std::vector<Fracture> allFractures = mergeAllFractures(internalFractures, externalFractures);

    std::vector<GroundTruthBox> groundTruths = loadGroundTruthBoxes(imagePath, grayEnhanced.size(), verboseOutput);
    if (!groundTruths.empty()) {
        stats = evaluateDetectionsWithIoU(allFractures, groundTruths, IOU_THRESHOLD, verboseOutput);
    }
    else {
        if (verboseOutput) {
            std::cout << "Nessuna ground truth disponibile per questa immagine." << std::endl;
        }
    }

    if (verboseOutput) {
        std::cout << "Risultato immagine:" << std::endl;
        std::cout << "TP = " << stats.tp << std::endl;
        std::cout << "FP = " << stats.fp << std::endl;
        std::cout << "FN = " << stats.fn << std::endl;

        if ((stats.tp + stats.fn) > 0) {
            double sensitivity = static_cast<double>(stats.tp) / static_cast<double>(stats.tp + stats.fn);
            std::cout << "Sensitivity immagine = "
                << std::fixed << std::setprecision(4)
                << sensitivity
                << std::endl;
        }
        else {
            std::cout << "Sensitivity immagine non calcolabile: TP + FN = 0" << std::endl;
        }
    }

    if (showImages) {
        cv::Mat result;
        cv::cvtColor(grayEnhanced, result, cv::COLOR_GRAY2BGR);

        drawFractures(result, internalFractures, cv::Scalar(0, 255, 255), 3);
        drawFractures(result, externalFractures, cv::Scalar(0, 0, 255), 3);
        drawGroundTruthBoxes(result, groundTruths, cv::Scalar(255, 0, 255), 2);

        ucas::imshow("immagine originale", gray, false);
        ucas::imshow("Possibili fratture", result, true);
        cv::destroyAllWindows();
    }

    return stats;
}

void printProgressiveSensitivity(
    const EvaluationStats& totalStats,
    int processedImages,
    int totalImages,
    bool verboseOutput
) {
    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "SENSITIVITY CUMULATIVA DOPO "
        << processedImages
        << " / "
        << totalImages
        << " IMMAGINI"
        << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    if (verboseOutput) {
        std::cout << "TP cumulativi = " << totalStats.tp << std::endl;
        std::cout << "FP cumulativi = " << totalStats.fp << std::endl;
        std::cout << "FN cumulativi = " << totalStats.fn << std::endl;
    }

    if ((totalStats.tp + totalStats.fn) > 0) {
        double sensitivity = static_cast<double>(totalStats.tp) /
            static_cast<double>(totalStats.tp + totalStats.fn);

        if (verboseOutput) {
            std::cout << "Sensitivity cumulativa = TP / (TP + FN)" << std::endl;
            std::cout << "Sensitivity cumulativa = "
                << totalStats.tp
                << " / ("
                << totalStats.tp
                << " + "
                << totalStats.fn
                << ")"
                << std::endl;
        }

        std::cout << "Sensitivity cumulativa = "
            << std::fixed << std::setprecision(4)
            << sensitivity
            << std::endl;

        std::cout << "Sensitivity cumulativa percentuale = "
            << std::fixed << std::setprecision(2)
            << sensitivity * 100.0
            << "%"
            << std::endl;
    }
    else {
        std::cout << "Sensitivity cumulativa non calcolabile: TP + FN = 0" << std::endl;
    }
}


void processSingleImageNormalMode(const std::string& imagePath) {
    int bitsUsed = 0;
    cv::Mat img = ucas::imread(imagePath, cv::IMREAD_UNCHANGED, &bitsUsed);
    if (img.empty()) {
        UCAS_THROW(ucas::strprintf("Impossibile aprire l'immagine: %s", imagePath.c_str()));
    }

    cv::Mat gray = improveImage(img, bitsUsed);

    cv::Mat boneMask;
    cv::Mat edges;
    computeSegmentation(gray, boneMask, edges);

    cv::Mat grayEnhanced = enhanceGrayForDisplay(gray, boneMask);

    std::vector<Fracture> externalFractures = detectExternalFractures(boneMask, edges);

    cv::Mat innerBoneMask;
    cv::Mat internalGradient;
    std::vector<InternalGradientDebug> debugInfos;
    std::vector<Fracture> internalFractures = detectInternalFracturesByGradientAndGray(
        grayEnhanced,
        boneMask,
        innerBoneMask,
        internalGradient,
        debugInfos
    );

    std::vector<Fracture> allFractures = mergeAllFractures(internalFractures, externalFractures);

    std::vector<GroundTruthBox> groundTruths = loadGroundTruthBoxes(imagePath, grayEnhanced.size(), true);
    if (!groundTruths.empty()) {
        evaluateDetectionsWithIoU(allFractures, groundTruths, IOU_THRESHOLD, true);
    }

    cv::Mat result;
    cv::cvtColor(grayEnhanced, result, cv::COLOR_GRAY2BGR);

    drawFractures(result, internalFractures, cv::Scalar(0, 255, 255), 3);
    drawFractures(result, externalFractures, cv::Scalar(0, 0, 255), 3);
    drawGroundTruthBoxes(result, groundTruths, cv::Scalar(255, 0, 255), 2);

    ucas::imshow("immagine originale", gray, false);
    ucas::imshow("Possibili fratture", result, true);
    cv::destroyAllWindows();
}

void runNormalModeWithoutSensitivity() {
    std::string imagePath;
    std::cout << "Inserisci il percorso dell'immagine: ";
    std::getline(std::cin, imagePath);

    processSingleImageNormalMode(imagePath);
}

void printFinalSensitivityResults(
    const EvaluationStats& totalStats,
    int totalImages,
    int processedImages,
    int skippedImages
) {
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "RISULTATI FINALI SU TUTTE LE IMMAGINI" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "Immagini trovate nella cartella: " << totalImages << std::endl;
    std::cout << "Immagini elaborate: " << processedImages << std::endl;
    std::cout << "Immagini saltate per errore: " << skippedImages << std::endl;

    std::cout << "TP totali = " << totalStats.tp << std::endl;
    std::cout << "FP totali = " << totalStats.fp << std::endl;
    std::cout << "FN totali = " << totalStats.fn << std::endl;

    if ((totalStats.tp + totalStats.fn) > 0) {
        double sensitivity = static_cast<double>(totalStats.tp) /
            static_cast<double>(totalStats.tp + totalStats.fn);

        std::cout << std::endl;
        std::cout << "Sensitivity totale = TP / (TP + FN)" << std::endl;
        std::cout << "Sensitivity totale = "
            << totalStats.tp
            << " / ("
            << totalStats.tp
            << " + "
            << totalStats.fn
            << ")"
            << std::endl;

        std::cout << "Sensitivity totale = "
            << std::fixed << std::setprecision(4)
            << sensitivity
            << std::endl;

        std::cout << "Sensitivity percentuale = "
            << std::fixed << std::setprecision(2)
            << sensitivity * 100.0
            << "%"
            << std::endl;
    }
    else {
        std::cout << std::endl;
        std::cout << "Sensitivity totale non calcolabile: TP + FN = 0" << std::endl;
    }
}

void runSensitivityMode() {
    std::string imagesDir;
    std::cout << "Inserisci il percorso della cartella delle immagini: ";
    std::getline(std::cin, imagesDir);

    std::cout << std::endl;
    std::cout << "Scegli come vuoi eseguire la valutazione:" << std::endl;
    std::cout << "1 = mostra le immagini una alla volta e calcola la sensitivity" << std::endl;
    std::cout << "2 = automatico, senza mostrare immagini, calcola solo la sensitivity" << std::endl;
    std::cout << "Premi 1 oppure 2 e poi INVIO: ";

    std::string mode;
    std::getline(std::cin, mode);

    bool showImages = true;
    bool verboseOutput = true;

    if (!mode.empty() && mode[0] == '2') {
        showImages = false;
        verboseOutput = true;
    }

    EvaluationStats totalStats;
    int processedImages = 0;
    int skippedImages = 0;

    std::vector<cv::String> imagePaths = collectImagePaths(imagesDir);

    if (imagePaths.empty()) {
        std::cout << "Nessuna immagine trovata nella cartella:" << std::endl;
        std::cout << imagesDir << std::endl;
        std::cout << "Controlla che il percorso sia corretto e che le immagini abbiano estensione .png, .jpg, .jpeg, .bmp, .tif o .tiff." << std::endl;
        return;
    }

    std::cout << "Numero immagini trovate: " << imagePaths.size() << std::endl;
    if (showImages) {
        std::cout << "Le immagini verranno mostrate una alla volta." << std::endl;
        std::cout << "Chiudi la finestra oppure premi un tasto nella finestra per passare alla successiva." << std::endl;
    }
    else {
        std::cout << "Modalita automatica attiva: nessuna immagine verra' mostrata." << std::endl;
        std::cout << "Verranno stampate solo la sensitivity progressiva e quella finale." << std::endl;
    }

    for (size_t i = 0; i < imagePaths.size(); ++i) {
        std::string imagePath = imagePaths[i];

        try {
            EvaluationStats imageStats = processSingleImage(imagePath, showImages, verboseOutput);

            totalStats.tp += imageStats.tp;
            totalStats.fp += imageStats.fp;
            totalStats.fn += imageStats.fn;

            processedImages++;

            printProgressiveSensitivity(
                totalStats,
                processedImages,
                static_cast<int>(imagePaths.size()),
                verboseOutput
            );
        }
        catch (const std::exception& e) {
            std::cout << std::endl;
            std::cout << "Errore durante l'elaborazione di: " << imagePath << std::endl;
            std::cout << "Messaggio errore: " << e.what() << std::endl;
            skippedImages++;
        }
        catch (...) {
            std::cout << std::endl;
            std::cout << "Errore sconosciuto durante l'elaborazione di: " << imagePath << std::endl;
            skippedImages++;
        }
    }

    printFinalSensitivityResults(
        totalStats,
        static_cast<int>(imagePaths.size()),
        processedImages,
        skippedImages
    );
}
