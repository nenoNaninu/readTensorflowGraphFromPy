#include<Python.h>
#include<iostream>
#include<opencv2\opencv.hpp>
#include"LibHeader.h"
#include<Windows.h>

using namespace cv;
using std::cout;

int tensorflow(char* name) {

	Mat mat = imread(name);
	Mat img = Mat(28, 28, CV_8UC3);
	PyObject *session, *pArgs, *pModule, *pName, *sessGetFunc, *sessCloseFunc, *pImgPtr, *classifiedValue, *estimateFunc, *nullArg;
	PyObject *sampleFunc, *sampleValue;
	PyObject* createImgFunc;
	Py_Initialize();

	pName = PyUnicode_DecodeFSDefault("load_graph_output_y");

	pModule = PyImport_Import(pName);

	Py_DECREF(pName);

	if (pModule != NULL)
	{
		sessGetFunc = PyObject_GetAttrString(pModule, "getSession");

		if (sessGetFunc && PyCallable_Check(sessGetFunc))
		{
			//sessionが作られた。
			session = PyObject_CallObject(sessGetFunc, NULL);
			if (session != NULL)
			{
				std::cout << "session is not NULL" << std::endl;
				//printf("Result of call: %ld\n", PyLong_AsLong(session));
				//return -1;
			}
			else
			{
				std::cout << "session is NULL" << std::endl;
				return -1;
			}
		}
		else
		{
			cout << "can't get getSession" << std::endl;
			Py_DECREF(sessGetFunc);
			Py_DECREF(pModule);
			PyErr_Print();
			std::cout << "err" << std::endl;
			return -1;
		}

		createImgFunc = PyObject_GetAttrString(pModule, "createImgArray");

		if (!(createImgFunc && PyCallable_Check(createImgFunc)))
		{
			std::cout << "careateImgFunc can't call" << std::endl;
			return -1;
		}
		Py_buffer pyBuf;
		cout << "test" << std::endl;
		PyObject* tmp = PyObject_CallObject(createImgFunc, NULL);
		int result = PyObject_GetBuffer(tmp, &pyBuf, -1);
		if (result == -1) {
			cout << "cannot get buf" << std::endl;
			return -1;
		}
		float* bufptr = (float*)pyBuf.buf;
		char* imgPtr = (char*)img.data;

		estimateFunc = PyObject_GetAttrString(pModule, "estimate");

		if (!(estimateFunc && PyCallable_Check(estimateFunc)))
		{
			std::cout << "estimate can't call" << std::endl;
			return -1;
		}

		pArgs = PyTuple_New(2);
		PyTuple_SetItem(pArgs, 0, tmp);
		PyTuple_SetItem(pArgs, 1, session);
		
		//memcpy(pyBuf.buf,img.data,28*28*3*sizeof(float));
		//メモリコピーのルーチン。
		for (int i = 0; i < 28 * 28 * 3; i++) {
			bufptr[i] = imgPtr[i]/ 255.0;
		}


		//推測するためグラフに流し込む。
		unsigned long time = timeGetTime();
		classifiedValue = PyObject_CallObject(estimateFunc, pArgs);
		std::cout << "time" << (unsigned long)timeGetTime() - time << std::endl;
		long answer = PyLong_AsLong(classifiedValue);
		std::cout << "answer is" << answer << std::endl;
		sessCloseFunc = PyObject_GetAttrString(pModule, "close_sess");
		PyObject_CallObject(sessCloseFunc, pArgs);
	}
	else
	{
		PyErr_Print();
		fprintf(stderr, "Failed to load module");
		return 1;
	}

	Py_DECREF(createImgFunc);
	Py_DECREF(estimateFunc);
	Py_DECREF(sessGetFunc);
	Py_DECREF(pModule);
	return 0;
}

int main(int argc ,char* argv[]) {
	tensorflow(argv[1]);
	std::getchar();
	return 0;
}