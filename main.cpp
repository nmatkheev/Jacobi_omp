
#include <iostream>
#include <fstream>

#include <cmath>
#include <omp.h>

#include <vector>
#include <array>
#include <algorithm>

const double EPS = 0.0000001;


using namespace std;

//  Template alias - C++11
//template <size_t LEN>

//  Upgraded typedef - C++11
    using Matrix1D = std::vector<double>;

//template <class T, size_t ROW, size_t COL>
    using Matrix2D = std::vector<std::vector<double> >;




class LinearSystem
{
    public:
	    int dimension;
        Matrix2D A;
        Matrix1D F;
        Matrix1D X;
	    double time_simp;
	    double time_omp;

        void load_data();
        void solve_casual();
        void solve_omp();
        void init_data();
        void display_1d(Matrix1D);
        void display_2d(Matrix2D);

	LinearSystem(int N) {
        srand(time(0));
        dimension = N;

        Matrix2D A1(dimension);
        A = A1;
        Matrix1D F1(dimension);
        F = F1;
        Matrix1D X1(dimension);
        X = X1;


        A.resize(N);
        for (size_t i = 0; i<N; i++) {
            A[i].resize(N);
//            cout << "Init of " << i << " row in A" << endl;
        }
        F.resize(N);
//        cout << "resized F" << endl;
        X.resize(N);
//        cout << "resized X" << endl;

        time_omp = 0;
		time_simp = 0;

        cout << "Constructor is ok!" << endl;
	}

//	~LinearSystem(void)


};


void LinearSystem::load_data()
{
    cout << "Alive";
	for (int i = 0; i < dimension; i++) {
//        cout << "Alive entry Load_Data" << i << endl;

		for (int j = 0; j < dimension; j++) {
//            cout << "I alive" << j << endl;
			if (i == j)
				A[i][j] = 100 * dimension + rand() % 300 * dimension;
			else
				A[i][j] = 1 + rand() % 100;
		}
		F[i] = 1 + rand() % 10;
		//cin >> F[i];
		X[i] = 1;
	}
}


void LinearSystem::display_1d(Matrix1D arg)
{
	for (size_t i = 0; i < dimension; i++)
		cout << arg[i] << endl;
	cout << "=========================";
}


void LinearSystem::display_2d(Matrix2D arg)
{
	for (size_t i = 0; i < dimension; i++) {
		for (int g = 0; g < dimension; g++)
			cout << arg[i][g] << " ";
		cout << endl;
	}
	cout << endl;
}


/// N - размерность матрицы; A[N][N] - матрица коэффициентов, F[N] - столбец свободных членов,
/// X[N] - начальное приближение, также ответ записывается в X[N];
void LinearSystem::solve_casual()
{
	int N = dimension;
    int g;
	double time1, time2;

    Matrix1D TempX(N);

// Норма, определяемая как наибольшая разность столбца иксов соседней итерации
	double norm;

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
	}
    while (norm > EPS);

	time2 = omp_get_wtime();
	//display_data(X);
	time_simp = time2 - time1;
}


void LinearSystem::solve_omp()
{
	int N = dimension;
    int g;
	double t1, t2;

    Matrix1D TempX(N);

    double norm;
	t1 = omp_get_wtime();

	do
    {
    #pragma omp parallel for private(g) shared(TempX)
		for (int i = 0; i < N; i++)
        {
			TempX[i] = F[i];
			for (g = 0; g < N; g++)
            {
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
	}
    while (norm > EPS);

	t2 = omp_get_wtime();
	//display_data(X);
	time_omp = t2 - t1;
}


void LinearSystem::init_data() {
	for (size_t i = 0; i < dimension; i++)
		X[i] = 1;
}



int main(int argc, char* argv[])
{
	int num; // = atoi(argv[2]);
	cin >> num;
	int dimension; //= atoi(argv[1]);
	cin >> dimension;
 	omp_set_num_threads(num);
    cout << "I started\n";
	LinearSystem *ls = new LinearSystem(dimension);

    ls->load_data();


    ls->solve_omp();

    ls->init_data();
    ls->solve_casual();



    cout << "Casual time: " << ls->time_simp << endl;
    cout << "OpenMP time: " << ls->time_omp << endl;


    return 0;
}
