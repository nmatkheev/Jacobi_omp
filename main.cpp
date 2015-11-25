// Jacobi.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <string.h>

using namespace std;

const double eps = 0.0000001;

class LinearSystem
{
private:
	int Dimension;
	double** A;
	double* F;
	double* X;
public:
	double time_simp;
	double time_omp;
	LinearSystem(int N)
	{
		srand(time(0));
		Dimension = N;
		A = new double*[N];
		F = new double[N];
		X = new double[N];
		time_omp = 0;
		time_simp = 0;
		this->LoadData();
	}
	~LinearSystem(void)
	{
		delete A;
		delete F;
		delete X;
	}
	void LoadData(void);
	void CasualSolve(void);
	void OpenMPSolve(void);
	void ClearX(void);
};

void LinearSystem::LoadData(void)
{

	for (int i = 0; i < Dimension; i++) {
		A[i] = new double[Dimension];
		for (int j = 0; j < Dimension; j++) {
			if (i == j)
				A[i][j] = 100 * Dimension + rand() % 300 * Dimension;
			else
				A[i][j] = 1 + rand() % 100;
		}
		F[i] = 1 + rand() % 10;
		//cin >> F[i];
		X[i] = 1;
	}
}

void LinearSystem::DisplayData(double* arg) {
	for (int i = 0; i < Dimension; i++)
		cout << arg[i] << endl;
	cout << "=========================";
}

void LinearSystem::DisplayData(double** arg) {
	for (int i = 0; i < Dimension; i++) {
		for (int g = 0; g < Dimension; g++)
			cout << arg[i][g] << " ";
		cout << endl;
	}
	cout << endl;
}

/// N - ðàçìåðíîñòü ìàòðèöû; A[N][N] - ìàòðèöà êîýôôèöèåíòîâ, F[N] - ñòîëáåö ñâîáîäíûõ ÷ëåíîâ,
/// X[N] - íà÷àëüíîå ïðèáëèæåíèå, îòâåò çàïèñûâàåòñÿ òàêæå â X[N];
void LinearSystem::CasualSolve(void)
{
	int N = Dimension, g;
	double time1, time2; //Äëÿ çàìåðà âðåìåíè
	double* TempX = new double[N];

	double* F1 = new double[N];
	double* X1 = new double[N];
	double norm; // íîðìà, îïðåäåëÿåìàÿ êàê íàèáîëüøàÿ ðàçíîñòü êîìïîíåíò ñòîëáöà èêñîâ ñîñåäíèõ èòåðàöèé.

	//------------------------------------------------NO OpenMP begins--------------------------------------------------//
	memcpy(F1, F, sizeof(F));
	memcpy(X1, X, sizeof(X));
	time1 = omp_get_wtime();
	do {
		for (int i = 0; i < N; i++) {
			TempX[i] = F[i];
			for (g = 0; g < N; g++) {
				if (i != g)
					TempX[i] -= A[i][g] * X[g];
			}
			TempX[i] /= A[i][i];
		}
		norm = fabs(X[0] - TempX[0]);
		for (int h = 0; h < N; h++) {
			if (fabs(X[h] - TempX[h]) > norm)
				norm = fabs(X[h] - TempX[h]);
			X[h] = TempX[h];
		}
	} while (norm > eps);
	time2 = omp_get_wtime();
	//DisplayData(X);
	//cout << "Time elapsed (simple): " << time2 - time1 << endl;
	time_simp = time2 - time1;
	//-----------------------------------------------Ending NO OpenMP-----------------------------------------------------//
	delete F1;
	delete TempX;
	delete X1;
}

void LinearSystem::OpenMPSolve(void) {
	int N = Dimension, g;
	double t1, t2; //Äëÿ çàìåðà âðåìåíè
	double* TempX = new double[N];

	double* F1 = new double[N];
	double* X1 = new double[N];
	double norm; // íîðìà, îïðåäåëÿåìàÿ êàê íàèáîëüøàÿ ðàçíîñòü êîìïîíåíò ñòîëáöà èêñîâ ñîñåäíèõ èòåðàöèé.
	//------------------------------------------------OpenMP begins-------------------------------------------------------//
	memcpy(F1, F, sizeof(F));
	memcpy(X1, X, sizeof(X));
	t1 = omp_get_wtime();
	do {
		///////////////////////////////////////////////////////////////////////////
#pragma omp parallel for private(g) shared(TempX,F1,X1) 
		for (int i = 0; i < N; i++) {
			TempX[i] = F[i];
			for (g = 0; g < N; g++) {
				if (i != g)
					TempX[i] -= A[i][g] * X[g];
			}
			TempX[i] /= A[i][i];
		}
		//////////////////////////////////////////////////////////////////////////
		norm = fabs(X[0] - TempX[0]);
		for (int h = 0; h < N; h++) {
			if (fabs(X[h] - TempX[h]) > norm)
				norm = fabs(X[h] - TempX[h]);
			X[h] = TempX[h];
		}
	} while (norm > eps);
	t2 = omp_get_wtime();
	//DisplayData(X);
	//cout << "Time elapsed (openMP): " << t2 - t1 << endl;
	time_omp = t2 - t1;
	//-----------------------------------------------Ending OpenMP-----------------------------------------------------//
}

void LinearSystem::ClearX(void) {
	for (int i = 0; i < Dimension; i++)
		X[i] = 1;
}

int main(int argc, char* argv[])
{
	int num; // = atoi(argv[2]);
	cin >> num;
	int dimension; //= atoi(argv[1]);
	cin >> dimension;
	omp_set_num_threads(num);
	LinearSystem* ls = new LinearSystem(dimension);
	ls->OpenMPSolve();
	ls->ClearX();
	ls->CasualSolve();
	cout << "OpenMP time: " << ls->time_omp << endl << "Casual time: " << ls->time_simp;
	return 0;
}
