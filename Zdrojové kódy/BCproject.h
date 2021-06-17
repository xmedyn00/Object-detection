#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_BCproject.h"

#include <QMainWindow>
#include <QDebug>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QImage>
#include <QPixmap>
#include <QCloseEvent>
#include <QMessageBox>
#include <QFileDialog>

#include <fstream>
#include <sstream>
#include <iostream>

#include "opencv2/opencv.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class BCproject : public QMainWindow
{
    Q_OBJECT

protected:
	void PlayVideo();
	void closeEvent(QCloseEvent *event);
public:
	BCproject(QWidget *parent = Q_NULLPTR);
private slots:
	// Odstraneni nepotrebnych bboxu metodou non-maxima suppression
	void postprocess(Mat& frame, const vector<Mat>& out);
	// Kresleni bboxu
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
	// Ziskani nazvu vystupnich vrstev
	vector<String> getOutputsNames(const Net& net);
	
	//tlacitka k ovladani prehravace
	void on_playBtn_clicked();
	void on_StopBtn_clicked();
	void on_LoadVideoBtn_clicked();
	void on_PauseBtn_clicked();
	void on_chooseVideoBtn_clicked();
	void on_realTimeBtn_clicked();
	//tlacitka pro objekty detekce
	void on_BtnCar_clicked();
	void on_BtnPerson_clicked();
	void on_BtnTruck_clicked();
	void on_BtnVan_clicked();
	void on_BtnBicycle_clicked();
	void on_BtnMotorbike_clicked();
	void on_BtnBus_clicked();
	void on_BtnDetect_clicked();
	//tlacitka pro ROI
	void on_BtnROI_clicked();
	void on_BtnROIup_pressed();
	void on_BtnROIdown_pressed();
	void on_BtnROIright_pressed();
	void on_BtnROIleft_pressed();
	void on_BtnROIzoomPlus_pressed();
	void on_BtnROIzoomMinus_pressed();
	//kontrol barvy tlacitek
	void btnColorCheck();

	void on_slider_sliderMoved(int position);
	

private:
	Ui::BCprojectClass ui;
	QGraphicsPixmapItem pixmap;
	cv::VideoCapture cap;
	

	float confThreshold = 0.5;	// Prahova hodnota spolehlivosti 
	float nmsThreshold = 0.4;	// Prahova hodnota maximalniho potlaceni
	int inpWidth = 416;			// Sirka vstupniho obrazu site
	int inpHeight = 416;		// Vyska vstupniho obrazu site

	// Nacitani nazvu trid
	string classesFile = "obj.names";
	string line;
	// Konfiguracni a vahove soubory pro model detektoru
	//soubory pro yolov4
	String modelConfiguration = "yolov4-obj.cfg";
	String modelWeights = "yolov4-obj_best.weights";
	//soubory pro yolov4-tiny
	//String modelConfiguration = "yolov4-tiny-obj.cfg";
	//String modelWeights = "yolov4-tiny-obj_best.weights";
	
	// Nacitani site
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	Mat frame, blob, overlay;
	
	//promena kam se ulozi vybrane video
	QString dirOpen;
	//cislo atkuatlniho frajmu
	int framecount_get = 0;

	

	vector<string> classes;
	
	//pomoce promeny k rizeni objektu detekci
	bool detec_car = true;
	bool detec_person = true;
	bool detec_truck = true;
	bool detec_van = true;
	bool detec_bicycle = true;
	bool detec_motorbike = true;
	bool detec_bus = true;
	
	//Promeny ROI
	bool isSelectedRoiArea;
	int posXroi = 100;
	int posYroi = 100;
	int sizeWroi = 500;
	int sizeHroi = 500;
	

	//pokud streamType = 0 video vybereme s pocitacu nebo komboboxu, pokud streamType = 1 jsme v rezimu realtime a muzeme spustit video s kamery
	int streamType = 0;

	
};
