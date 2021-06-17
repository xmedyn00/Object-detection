#include "BCproject.h"


BCproject::BCproject(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	ui.graphicsView->setScene(new QGraphicsScene(this));
	ui.graphicsView->scene()->addItem(&pixmap);

	// scitavi nazvu trid
	ifstream ifs(classesFile.c_str());
	while (getline(ifs, line)) classes.push_back(line);

	// Nastaveni site
	net.setPreferableBackend(DNN_BACKEND_CUDA);
	net.setPreferableTarget(DNN_TARGET_CUDA);

	//combobox config
	ui.comboBox->addItem("Choose video...");
	ui.comboBox->addItem("test_video.mp4");

	if (modelConfiguration == "yolov4-obj.cfg") {
		ui.label_18->setText("yolov4");
	}
	else if (modelConfiguration == "yolov4-tiny-obj.cfg") {
		ui.label_18->setText("yolov4-tiny");
	}

	ui.label_15->setText(CV_VERSION);

}


//Tlacitko pro otevreni videa
void BCproject::on_LoadVideoBtn_clicked() {
	QString comboBoxValue = ui.comboBox->currentText();
	if (comboBoxValue == "Choose video...") {	//pokud kombox nastaven na "Choose video..." tak je moznost vybrat video na pocitaci
		dirOpen = QFileDialog::getOpenFileName(this, tr("Open File"), "/", tr("Video (*.mp4)"));
	}
	else {		//jinak muzeme spustit video z komboboxu
		dirOpen = ui.comboBox->currentText();
	}

	if (dirOpen.isEmpty())
	{
		QMessageBox msgBox;
		msgBox.setText("The file was not selected!");
		msgBox.exec();
	}
	else {
		framecount_get = 0;		//vynulovani frajmu
		PlayVideo();	//spousteni videa
	}

}

//pohyb slajderu
void BCproject::on_slider_sliderMoved(int position)
{
	cap.set(CAP_PROP_POS_FRAMES, position);
}


//Metoda pro spousteni videa a zpracovani kazdeho snimku
void BCproject::PlayVideo() {
	if (streamType == 0) {		//tlacitko choose video nastavi hodnotu promena streamType na 0 a tim povoli spousteni videa s pocitace 
		cap.open(QString("%1").arg(dirOpen).toStdString());
	}
	else if (streamType == 1) {		//tlacitko real time nastavi hodnotu promene streamType na 1 a otevre video s pripojene kamery
		cap.open(0);
	}

	cap.set(CAP_PROP_POS_FRAMES, framecount_get);	//video se otevre s cisla frajmu uvedeneho v promene framecount_get

	//nastaveni slajderu
	int contFrames = cap.get(CAP_PROP_FRAME_COUNT);
	ui.slider->setMinimum(0);
	ui.slider->setMaximum(contFrames);

	//overeni pripojeni kamery
	if (streamType == 1) {
		if (!cap.isOpened()) {
			QMessageBox msgBox;
			msgBox.setText("Camera is not connected!");
			msgBox.exec();
		}
	}
	int lastFrame = cap.get(CAP_PROP_FRAME_COUNT);
	while (1)
	{
		ui.slider->setSliderPosition(cap.get(CAP_PROP_POS_FRAMES));

		cap >> frame;  //ziskame snimek z videa
		
		if (lastFrame == cap.get(CAP_PROP_POS_FRAMES)) {
			on_StopBtn_clicked();
		}
		//zastavit program pokud dojde ke koncí videa
		if (frame.empty()) {
			break;
		}

		//po nacteni snimku z videa nebo kamery on je preveden pres f-ci blobFromImage na objekt blob pro neuronovou sit. 
		//Na vystupu teto f-ci dostame obraz velikosti (416, 416)
		blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

		//Nastavi blob na vstup do siti
		net.setInput(blob);

		// Spusti pruchod vpred pro vypocet vystupu vrstvy se jmenem objektu detekci
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));

		// odstranime ohranicujici ramecky s nizkou jistotou
		postprocess(frame, outs);

		// Doplnujici informace. 
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		double fps = 1 / (t / 1000);
		ui.label_8->setText(QString("%1").arg(fps));
		//ui.label_18->setText("sdasd");

		// Zpracovani obrazu v rezimu ROI
		if (isSelectedRoiArea) {
			float alpha = 0.5;
			frame.copyTo(overlay);
			rectangle(frame, Rect(posXroi, posYroi, sizeWroi, sizeHroi), Scalar(100, 255, 100), -1);
			addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame, 0);
		}

		// Vypis frajmu s bboxy
		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);
		if (!frame.empty())
		{
			//aby svravne zobrazit obrazek Mat na Qt QGraphicsView je potreba prevest Mat na QImage. 
			//barevni prostor OpenCV je BGR a je potreba ho zmenit na RGB, potom QImage prevedeme na QPixmap
			//nakonec pridavame pixmap do QGraphicsScene
			QImage qimg(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
			pixmap.setPixmap(QPixmap::fromImage(qimg.rgbSwapped()));
			// aby odrazek vzdy hodil k zobrzeni bez ohledu na velikost okna aplikace
			ui.graphicsView->fitInView(&pixmap, Qt::KeepAspectRatio);

			qApp->processEvents();
		}
	}
}

//podminka zavreni aplikace
void BCproject::closeEvent(QCloseEvent *event)
{
	if (cap.isOpened())
	{
		QMessageBox::warning(this,
			"Warning",
			"Please stop the video before closing the application!");
		event->ignore();
	}
	else
	{
		event->accept();
	}
}

//tlacitka k ovladani prehravace
//Play
void BCproject::on_playBtn_clicked()
{
	PlayVideo();
}
//Stop
void BCproject::on_StopBtn_clicked() {
	framecount_get = 0;
	cap.release();
}
//Pause
void BCproject::on_PauseBtn_clicked() {
	framecount_get = cap.get(CAP_PROP_POS_FRAMES);	
	cap.release();
}

//rezim vyberu videa
void BCproject::on_chooseVideoBtn_clicked() {
	streamType = 0;
	ui.PauseBtn->setDisabled(0);
	ui.playBtn->setDisabled(0);
	ui.StopBtn->setDisabled(0);
	ui.comboBox->setDisabled(0);
	ui.LoadVideoBtn->setText("Open video");
	ui.realTimeBtn->setStyleSheet("background:rgb(207, 207, 207)");
	ui.chooseVideoBtn->setStyleSheet("background:rgba(100, 195, 35, 255)");
	ui.comboBox->clear();
	ui.comboBox->addItem("Choose video...");
	ui.comboBox->addItem("test_video.mp4");

}

//rezim Real time
void BCproject::on_realTimeBtn_clicked() {
	//cap.release();
	streamType = 1;
	ui.PauseBtn->setDisabled(1);
	ui.playBtn->setDisabled(1);
	ui.StopBtn->setDisabled(0);
	ui.comboBox->setDisabled(1);
	ui.LoadVideoBtn->setText("Start steaming");
	ui.realTimeBtn->setStyleSheet("background:rgba(100, 195, 35, 255)");
	ui.chooseVideoBtn->setStyleSheet("background:rgb(207, 207, 207)");
	ui.comboBox->clear();
	ui.comboBox->addItem("Streaming...");
}

//tlacitka pro rizeni objektu detekce
//auto
void BCproject::on_BtnCar_clicked() {
	if (!detec_car) {
		detec_car = true;
		ui.BtnCar->setText("ON");
	}
	else {
		detec_car = false;
		ui.BtnCar->setText("OFF");
	}
	btnColorCheck();

}
//clovek
void BCproject::on_BtnPerson_clicked() {
	if (!detec_person) {
		detec_person = true;
		ui.BtnPerson->setText("ON");
	}

	else {
		detec_person = false;
		ui.BtnPerson->setText("OFF");
	}
	btnColorCheck();
}
//kamion
void BCproject::on_BtnTruck_clicked() {
	if (!detec_truck) {
		detec_truck = true;
		ui.BtnTruck->setText("ON");
	}

	else {
		detec_truck = false;
		ui.BtnTruck->setText("OFF");
	}
	btnColorCheck();
}
//dodavka
void BCproject::on_BtnVan_clicked() {
	if (!detec_van) {
		detec_van = true;
		ui.BtnVan->setText("ON");
	}

	else {
		detec_van = false;
		ui.BtnVan->setText("OFF");
	}
	btnColorCheck();
}
//kolo
void BCproject::on_BtnBicycle_clicked() {
	if (!detec_bicycle) {
		detec_bicycle = true;
		ui.BtnBicycle->setText("ON");
	}

	else {
		detec_bicycle = false;
		ui.BtnBicycle->setText("OFF");
	}
	btnColorCheck();
}
//motorka
void BCproject::on_BtnMotorbike_clicked() {
	if (!detec_motorbike) {
		detec_motorbike = true;
		ui.BtnMotorbike->setText("ON");
	}

	else {
		detec_motorbike = false;
		ui.BtnMotorbike->setText("OFF");
	}
	btnColorCheck();
}
//autobus
void BCproject::on_BtnBus_clicked() {
	if (!detec_bus) {
		detec_bus = true;
		ui.BtnBus->setText("ON");
	}

	else {
		detec_bus = false;
		ui.BtnBus->setText("OFF");
	}
	btnColorCheck();
}
//detekovat vsechny objekty
void BCproject::on_BtnDetect_clicked() {
	detec_motorbike = true;
	detec_bicycle = true;
	detec_van = true;
	detec_truck = true;
	detec_person = true;
	detec_car = true;
	detec_bus = true;
	btnColorCheck();
}

//kontroluje barvu tlacitek
void BCproject::btnColorCheck() {
	QString textBtnMotorbike = ui.BtnMotorbike->text();
	QString textBtnBicycle = ui.BtnBicycle->text();
	QString textBtnVan = ui.BtnVan->text();
	QString textBtnTruck = ui.BtnTruck->text();
	QString textBtnPerson = ui.BtnPerson->text();
	QString textBtnCar = ui.BtnCar->text();
	QString textBtnBus = ui.BtnBus->text();

	if (textBtnMotorbike == "ON") {
		ui.BtnMotorbike->setStyleSheet("background:rgba(100, 195, 35, 255)");
	}
	else {
		ui.BtnMotorbike->setStyleSheet("background:rgb(207, 207, 207)");
	}

	if (textBtnBicycle == "ON") {
		ui.BtnBicycle->setStyleSheet("background:rgba(100, 195, 35, 255)");
	}
	else {
		ui.BtnBicycle->setStyleSheet("background:rgb(207, 207, 207)");
	}

	if (textBtnVan == "ON") {
		ui.BtnVan->setStyleSheet("background:rgba(100, 195, 35, 255)");
	}
	else {
		ui.BtnVan->setStyleSheet("background:rgb(207, 207, 207)");
	}

	if (textBtnTruck == "ON") {
		ui.BtnTruck->setStyleSheet("background:rgba(100, 195, 35, 255)");
	}
	else {
		ui.BtnTruck->setStyleSheet("background:rgb(207, 207, 207)");
	}

	if (textBtnPerson == "ON") {
		ui.BtnPerson->setStyleSheet("background:rgba(100, 195, 35, 255)");
	}
	else {
		ui.BtnPerson->setStyleSheet("background:rgb(207, 207, 207)");
	}

	if (textBtnCar == "ON") {
		ui.BtnCar->setStyleSheet("background:rgba(100, 195, 35, 255)");
	}
	else {
		ui.BtnCar->setStyleSheet("background:rgb(207, 207, 207)");
	}

	if (textBtnBus == "ON") {
		ui.BtnBus->setStyleSheet("background:rgba(100, 195, 35, 255)");
	}
	else {
		ui.BtnBus->setStyleSheet("background:rgb(207, 207, 207)");
	}

	if (detec_motorbike == true && detec_bicycle == true && detec_van == true && detec_truck == true && detec_person == true && detec_car == true && detec_bus == true) {
		ui.BtnDetect->setStyleSheet("background:rgba(100, 195, 35, 255)");
		ui.BtnMotorbike->setText("ON");
		ui.BtnMotorbike->setStyleSheet("background:rgba(100, 195, 35, 255)");
		ui.BtnBicycle->setText("ON");
		ui.BtnBicycle->setStyleSheet("background:rgba(100, 195, 35, 255)");
		ui.BtnVan->setText("ON");
		ui.BtnVan->setStyleSheet("background:rgba(100, 195, 35, 255)");
		ui.BtnTruck->setText("ON");
		ui.BtnTruck->setStyleSheet("background:rgba(100, 195, 35, 255)");
		ui.BtnPerson->setText("ON");
		ui.BtnPerson->setStyleSheet("background:rgba(100, 195, 35, 255)");
		ui.BtnCar->setText("ON");
		ui.BtnCar->setStyleSheet("background:rgba(100, 195, 35, 255)");
		ui.BtnBus->setText("ON");
		ui.BtnBus->setStyleSheet("background:rgba(100, 195, 35, 255)");
	}
	else {
		ui.BtnDetect->setStyleSheet("background:rgb(207, 207, 207)");
	}


}

// Tlacitka pro ovladani ROI
//zapnout ROI
void BCproject::on_BtnROI_clicked() {
	posXroi = frame.size().width / 2 - sizeWroi / 2;
	posYroi = frame.size().height / 2 - sizeWroi / 2;
	if (!isSelectedRoiArea) {
		isSelectedRoiArea = true;
		ui.BtnROI->setStyleSheet("background:rgba(100, 195, 35, 255)");
	}
	else {
		isSelectedRoiArea = false;
		ui.BtnROI->setStyleSheet("background:rgb(207, 207, 207)");
	}
	
}
//posunout ROI dolu
void BCproject::on_BtnROIdown_pressed() {
	posYroi += 10;
}
//posunout ROI vpravo
void BCproject::on_BtnROIright_pressed() {
	posXroi += 10;
}
//posunout ROI nahoru
void BCproject::on_BtnROIup_pressed() {
	posYroi -= 10;
}
//posunout ROI vlevo
void BCproject::on_BtnROIleft_pressed() {
	posXroi -= 10;
}
//zvetsit ROI oblast
void BCproject::on_BtnROIzoomPlus_pressed() {
	sizeWroi += 10;
	sizeHroi += 10; 
	posXroi -= 5;
	posYroi -= 5;
}
//zmensit ROI oblast
void BCproject::on_BtnROIzoomMinus_pressed() {
	sizeWroi -= 10;
	sizeHroi -= 10;
	posXroi += 5;
	posYroi += 5;
}

//Odstrani ohranicujici ramecky s nizkou jistotou
void BCproject::postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Ziskame hodnotu a umisteni maximalniho skore
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

	//Provedeme potlaceni maxima(non maximum suppression) aby odstanit prekryvajici bloky s nizsimi duvernostmi
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}

}

// Nakresli predpokladane ohranicovaci pole
void BCproject::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{

	//Ziska stitek s nazvem tridy a jeji duveryhodnost
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;

	}

	//Zobrazi stitek v horni casti ohranicujiciho ramecku
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_DUPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);


	if (!isSelectedRoiArea) {
		if (detec_car) {
			if (classId == 0) {
				rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
				if (ui.checkBox_2->isChecked()) {
					frame.copyTo(overlay);
					rectangle(overlay, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(100, 255, 100), -1);
					addWeighted(overlay, 0.5, frame, 1 - 0.5, 0, frame);
					putText(frame, label, Point(left, top), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 1);
				}
			}


		}
		if (detec_person) {
			if (classId == 1) {
				rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);
				if (ui.checkBox_2->isChecked()) {
					frame.copyTo(overlay);
					rectangle(overlay, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(100, 255, 100), -1);
					addWeighted(overlay, 0.5, frame, 1 - 0.5, 0, frame);
					putText(frame, label, Point(left, top), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 1);
				}
			}

		}
		if (detec_truck) {
			if (classId == 2) {
				rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 0, 0), 3);
				if (ui.checkBox_2->isChecked()) {
					frame.copyTo(overlay);
					rectangle(overlay, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(100, 255, 100), -1);
					addWeighted(overlay, 0.5, frame, 1 - 0.5, 0, frame);
					putText(frame, label, Point(left, top), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 1);
				}
			}

		}
		if (detec_van) {
			if (classId == 3) {
				rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 3);
				if (ui.checkBox_2->isChecked()) {
					frame.copyTo(overlay);
					rectangle(overlay, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(100, 255, 100), -1);
					addWeighted(overlay, 0.5, frame, 1 - 0.5, 0, frame);
					putText(frame, label, Point(left, top), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 1);
				}
			}

		}
		if (detec_bicycle) {
			if (classId == 4) {
				rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 233, 255), 3);
				if (ui.checkBox_2->isChecked()) {
					frame.copyTo(overlay);
					rectangle(overlay, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(100, 255, 100), -1);
					addWeighted(overlay, 0.5, frame, 1 - 0.5, 0, frame);
					putText(frame, label, Point(left, top), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 1);
				}
			}

		}
		if (detec_motorbike) {
			if (classId == 5) {
				rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 127, 0), 3);
				if (ui.checkBox_2->isChecked()) {
					frame.copyTo(overlay);
					rectangle(overlay, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(100, 255, 100), -1);
					addWeighted(overlay, 0.5, frame, 1 - 0.5, 0, frame);
					putText(frame, label, Point(left, top), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 1);
				}
			}

		}
		if (detec_bus) {
			if (classId == 6) {
				rectangle(frame, Point(left, top), Point(right, bottom), Scalar(97, 0, 255), 3);
				if (ui.checkBox_2->isChecked()) {
					frame.copyTo(overlay);
					rectangle(overlay, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(100, 255, 100), -1);
					addWeighted(overlay, 0.5, frame, 1 - 0.5, 0, frame);
					putText(frame, label, Point(left, top), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 1);
				}
			}
		}
	}
	if (isSelectedRoiArea) {
		if (right >= posXroi && left <= posXroi + sizeWroi && top >= posYroi && top <= posYroi + sizeHroi) { //right >= posXroi && left <= posXroi + sizeWroi && top >= posYroi && top <= posYroi + sizeHroi
			if (detec_car) {
				if (classId == 0) {
					rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
					if (ui.checkBox_2->isChecked()) {
						rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
						putText(frame, label, Point(left, top), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 1);
					}
				}


			}
			if (detec_person) {
				if (classId == 1) {
					rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);
					if (ui.checkBox_2->isChecked()) {
						rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
						putText(frame, label, Point(left, top), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 1);
					}
				}

			}
			if (detec_truck) {
				if (classId == 2) {
					rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 0, 0), 3);
					if (ui.checkBox_2->isChecked()) {
						rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
						putText(frame, label, Point(left, top), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 1);
					}
				}

			}
			if (detec_van) {
				if (classId == 3) {
					rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 3);
					if (ui.checkBox_2->isChecked()) {
						rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
						putText(frame, label, Point(left, top), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 1);
					}
				}

			}
			if (detec_bicycle) {
				if (classId == 4) {
					rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 233, 255), 3);
					if (ui.checkBox_2->isChecked()) {
						rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
						putText(frame, label, Point(left, top), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 1);
					}
				}

			}
			if (detec_motorbike) {
				if (classId == 5) {
					rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 127, 0), 3);
					if (ui.checkBox_2->isChecked()) {
						rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
						putText(frame, label, Point(left, top), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 1);
					}
				}

			}
			if (detec_bus) {
				if (classId == 6) {
					rectangle(frame, Point(left, top), Point(right, bottom), Scalar(97, 0, 255), 3);
					if (ui.checkBox_2->isChecked()) {
						rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
						putText(frame, label, Point(left, top), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 1);
					}
				}
			}
		}
	}



}

// Ziskam nazvy vystupnich vrstev
vector<String> BCproject::getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		// Ziskame indexy vystupnich vrstev, tj. vrstev s nepripojenymi vystupy
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//ziskame nazvy vsech vrstev v siti
		vector<String> layersNames = net.getLayerNames();

		// Ziskame nazvy vystupnich vrstev v nazvech
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}



