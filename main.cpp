#include <iostream>
#include <cmath>
#include <mpi/mpi.h>
#include <vector>

/*
 *
 */

typedef double coor_t;
typedef std::vector<coor_t> subarea_t;
typedef const coor_t (*func_t) (const coor_t, const coor_t);

const coor_t A1 = 0.0;
const coor_t A2 = 2.0;

struct Point {
	coor_t x;
	coor_t y;

	Point(){}
	Point(const coor_t init_x, const coor_t init_y): x(init_x), y(init_y) {}
};

inline const coor_t F(const coor_t x, const coor_t y) {
	coor_t x_plus_y_2 = (x + y) * (x + y);
	return 4 * (1 - 2 * x_plus_y_2) * exp(1 - x_plus_y_2);
}


inline const coor_t phi(const coor_t x, const coor_t y) {
	return exp(1 - (x + y) * (x + y));
}

inline const coor_t exact_solution(const coor_t x, const coor_t y) {
	return phi(x, y);
}


//inline coor_x(const int i) {
//
//}

class GlobalData {
public:
	const int N1;
	const int N2;

	const coor_t A1;
	const coor_t A2;
	const coor_t B1;
	const coor_t B2;

	const double q;

	const coor_t eps;

	inline coor_t f(const coor_t t) {
		return (pow(1 + t, q) - 1.0) / (pow(2, q) - 1);
	}

	inline coor_t x(const int idx) {
		coor_t t = static_cast<coor_t>(idx) / N1;
		return A2 * f(t) + A1 * (1 - f(t));
	}

	inline coor_t y(const int idx) {
		coor_t t = static_cast<coor_t>(idx) / N2;
		return B2 * f(t) + B1 * (1 - f(t));
	}

	inline coor_t hx(const int idx) {
		return 0.5 * (x(idx) + x(idx + 1));
	}

	inline coor_t hy(const int idx) {
		return 0.5 * (y(idx) + y(idx + 1));
	}
};


class LocalData: public GlobalData {
public:
	GlobalData gd;
	int rank;
	subarea_t p;
	subarea_t r;
	subarea_t tau;
	subarea_t g;

	LocalData(const GlobalData & gd, const int rank):
			GlobalData(gd), rank(rank), p(size()), r(size()), tau(size()), g(size())
	{
		for (int i = 0; i < width(); ++i) {
			for (int j = 0; j < height(); ++j) {
				put(p, i, j, phi(x(i), y(j)));
				put(r, i, j, - delta());
			}
		}
	}

	inline int width() {

	}

	inline int height() {

	}

	inline int size() {
		return width() * height();
	}

	inline const coor_t & get(const subarea_t & a, const int i, const int j) {
		return a[i * width() + j];
	}

	inline const coor_t & put(subarea_t & a, const int i, const int j, const coor_t v) {
		return a[i * width() + j] = v;
	}


	inline const coor_t delta(func_t func, const int i, const int j) {
		//   u
		// l c r
		//   d
		coor_t l = func(x(i-1), y(j));
		coor_t r = func(x(i+1), y(j));
		coor_t u = func(x(i), y(j-1));
		coor_t d = func(x(i), y(j+1));
		coor_t c = func(x(i),y(j));
		coor_t first = ( (c-l)/hx(i-1) - (r-c)/hx(i) ) / hx(i);
		coor_t second = ( (c-u)/hy(j-1) - (d-c)/hy(j) ) / hy(j);
		return - (first + second);
	}

};



int main(int argc, char * argv[]) {

	int err_code;
	err_code = MPI_Init(&argc, &argv);
	if (err_code) {
		return err_code;
	}



	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) {
		std::cout << "inited" << std::endl;
	}
	MPI_Finalize();
	return 0;
}