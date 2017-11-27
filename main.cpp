#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <utility>




typedef double coor_t;
typedef std::vector<coor_t> subarea_t;
typedef const coor_t (*func_t) (const coor_t, const coor_t);

const coor_t EPS = 0.00001;
const coor_t A1 = 0.0;
const coor_t A2 = 2.0;

//struct Point {
//	coor_t x;
//	coor_t y;
//
//	Point(){}
//	Point(const coor_t init_x, const coor_t init_y): x(init_x), y(init_y) {}
//};

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


class DeltaH {
public:
//	func_t func, const int i, const int j
//	DeltaH()

	static const coor_t compute(
			const coor_t c,
			const coor_t l, const coor_t r,
			const coor_t u, const coor_t d,
			const coor_t hi, const coor_t hi_1,
			const coor_t hj, const coor_t hj_1
	) {
		//   u
		// l c r
		//   d
		coor_t first = ( (c-l)/hi_1 - (r-c)/hi ) / hi;
		coor_t second = ( (c-u)/hj_1 - (d-c)/hj ) / hj;
		return - (first + second);
	}


};


//inline coor_x(const int i) {
//
//}

//class DeltaH {
//	const
//};


class OneDimensionData {
public:
	const int N;  // number of points
	const coor_t A;  // begin of segment
	const coor_t B;  // end of segment
	const coor_t q;  // eventually ratio
	const int min_idx;
	const int max_idx;

	OneDimensionData(const int N, const coor_t A, const coor_t B, const coor_t q,
	                 const int min_local_idx, const int max_local_idx) :
			N(N), A(A), B(B), q(q), min_idx(min_local_idx), max_idx(max_local_idx)
	{}

	inline coor_t f(const coor_t t) {
		return (pow(1 + t, q) - 1.0) / (pow(2, q) - 1);
	}

	inline coor_t coor(const int global_idx) {
		coor_t t = static_cast<coor_t>(global_idx) / N;
		return B * f(t) + A * (1 - f(t));
	}

	inline const coor_t h(const int global_idx) {
		return 0.5 * (coor(global_idx) + coor(global_idx + 1));
	}

//	inline bool is_max(const coor_t coor) {
//		return std::abs(coor - B) < EPS;
//	}
//
//	inline bool is_min(const coor_t coor) {
//		return std::abs(coor - A) < EPS;
//	}

	inline bool is_max(const int global_idx) {
		return global_idx == N;
	}

	inline bool is_min(const int global_idx) {
		return global_idx == 0;
	}

	inline bool is_border(const int global_idx) {
		return is_min(global_idx) or is_max(global_idx);
	}

	inline const int local(const int global_idx) {
		return global_idx - min_idx;
	}

	inline const int idx_count() {
		return max_idx - min_idx + 1;
	}

	inline const int side_processes_count() {
		return static_cast<int>(round(static_cast<double>(N) / idx_count()));
	}
};


class LocalProcess {
public:
	OneDimensionData x;
	OneDimensionData y;
	const coor_t eps;
	int rank;
	subarea_t p;
	subarea_t r;
//	subarea_t tau;
	subarea_t g;

	LocalProcess(const OneDimensionData & x_data, const OneDimensionData & y_data, const coor_t eps, const int rank):
			x(x_data), y(y_data), rank(rank), eps(eps), p(size()), r(size()), g(size()) //, tau(size())
	{
		for (int i = x.min_idx; i <= x.max_idx; ++i) {
			for (int j = y.min_idx; j <= y.max_idx; ++j) {
				if (x.is_border(i) or y.is_border(j)) {
					put(p, i, j, 0);  // can be any value
					put(r, i, j, 0);
				} else {
					coor_t deltah_p = DeltaH::compute(
							phi(x.coor(i), y.coor(j)),
							phi(x.coor(i - 1), y.coor(j)), phi(x.coor(i + 1), y.coor(j)),
							phi(x.coor(i), y.coor(j - 1)), phi(x.coor(i), y.coor(j + 1)),
							x.h(i), x.h(i - 1), y.h(j), y.h(j - 1)
					);
					put(p, i, j, phi(x.coor(i), y.coor(j)));
					put(r, i, j, -deltah_p - F(x.coor(i), y.coor(j)));
				}
				put(g, i, j, get(r, i, j));
			}
		}
	}

	void compute_tau() {
		// copy local vector 'r' for sending neighbors
		coor_t up[x.N], down[x.N], left[y.N], right[y.N];
		for (int i = x.min_idx; i <= x.max_idx; ++i) {
			if (x.is_border(i))
				continue;
			if (not y.is_border(y.min_idx))
				put(up, i, y.min_idx, get(r, i, y.min_idx));
			if (not y.is_border(y.max_idx))
				put(down, i, y.max_idx, get(r, i, y.max_idx));
		}
		for (int j = y.min_idx; j <= y.max_idx; ++j) {
			if (y.is_border(j))
				continue;
			if (not x.is_border(x.min_idx))
				put(left, x.min_idx, j, get(r, x.min_idx, j));
			if (not x.is_border(x.max_idx))
				put(right, x.max_idx, j, get(r, x.max_idx, j));
		}
		// send local vector 'r' to neighbors
		MPI_Request up_request, down_request, left_request, right_request;
		if (not y.is_border(y.min_idx))
			MPI_Isend(up, x.N, MPI_DOUBLE, rank - x.side_processes_count(), tag, MPI_COMM_WORLD, &up_request);
		if (not y.is_border(y.max_idx))
			MPI_Isend(down, x.N, MPI_DOUBLE, rank + x.side_processes_count(), tag, MPI_COMM_WORLD, &down_request);
		if (not x.is_border(x.min_idx))
			MPI_Isend(left, y.N, MPI_DOUBLE, rank - y.side_processes_count(), tag, MPI_COMM_WORLD, &left_request);
		if (not x.is_border(x.max_idx))
			MPI_Isend(right, y.N, MPI_DOUBLE, rank + y.side_processes_count(), tag, MPI_COMM_WORLD, &right_request);

		// TODO: compute local delta_r

		subarea_t delta_r(size());
		coor_t numerator = 0.0, denominator = 0.0;
		for (int i = x.min_idx; i <= x.max_idx; ++i) {
			if (x.is_border(i))
				continue;
			for (int j = y.min_idx; j <= y.max_idx; ++j) {
				if (y.is_border(j))
					continue;
				numerator += x.h(i) * y.h(j) * get(r, i, j) * get(r, i, j);
				denominator -= x.h(i) * y.h(j) * get(delta_r, i, j) * get(r, i, j);
			}
		}
		coor_t send_data[2], receive_data[2];
		send_data[0] = numerator;
		send_data[1] = denominator;
		MPI_Allreduce(send_data, receive_data, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	}

	inline int size() {
		return x.N * y.N;
	}

	inline const coor_t & get(const subarea_t & a, const int i, const int j) {
		return a[x.local(i) * x.N + y.local(j)];
	}

	inline const coor_t & put(subarea_t & a, const int i, const int j, const coor_t v) {
		return a[x.local(i) * x.N + y.local(j)] = v;
	}

	inline const coor_t & put(coor_t a[], const int i, const int j, const coor_t v) {
		return a[x.local(i) * x.N + y.local(j)] = v;
	}

	inline const coor_t & get(const coor_t a[], const int i, const int j) {
		return a[x.local(i) * x.N + y.local(j)];
	}

};


inline std::pair<int, int> compute_subfield_size(
		const int rank, const int n_side, const int count_indexes, bool on_side, const int row_length) {
	const int target_n = on_side ? rank % n_side : rank / row_length;
	int step = count_indexes / n_side;
	if (count_indexes % n_side == 0) {
		int min_local_idx = (count_indexes / n_side) * target_n;
		int max_local_idx = min_local_idx + step - 1;
		return std::make_pair(min_local_idx, max_local_idx);
	}
	int cur_n = n_side - 1;
	int max_local_idx = count_indexes - 1;
	// compute small fields
	while (cur_n != target_n and (max_local_idx + 1) % (step + 1) != 0) {
		--cur_n;
		max_local_idx -= step;
	}
	if ((max_local_idx + 1) % (step + 1) == 0) {
		++step;
		// compute big fields
		while (cur_n != target_n) {
			--cur_n;
			max_local_idx -= step;
		}
	}
	int min_local_idx = max_local_idx - step + 1;
	return std::make_pair(min_local_idx, max_local_idx);
}



int main(int argc, char * argv[]) {
	int process_count, rank;

	int err_code;
	err_code = MPI_Init(&argc, &argv);
	if (err_code) {
		return err_code;
	}



	const int N = 1000;


	// compute process on X and Y axes
	MPI_Comm_size(MPI_COMM_WORLD, &process_count);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	process_count = 12;

	std::cout << "process_count = " << process_count << std::endl;

	int a = static_cast<int>(sqrt(process_count));
	for (; process_count % a > 0; --a)
		;
	int b = process_count / a;
	// N == a * b
	// a - process number (X axes)
	// b - process number (Y axes)
	// compute size on X axes
//	std::cout << "a = " << a << std::endl;
//	std::cout << "b = " << b << std::endl;
//	std::cout << "hear" << std::endl;
	for (int i = 0; i < process_count; ++i) {
		rank = i;
		std::pair<int, int> x_range = compute_subfield_size(rank, a, N, true, a);
		std::pair<int, int> y_range = compute_subfield_size(rank, b, N, false, a);
		OneDimensionData x_data = OneDimensionData(N, 0, 2, 3 / 2, x_range.first, x_range.second);
		OneDimensionData y_data = OneDimensionData(N, 0, 2, 3 / 2, y_range.first, y_range.second);
//
		std::cout << rank << " x " << x_data.min_idx << '-' << x_data.max_idx
		          << "=" << x_data.idx_count()
		          << " y " << y_data.min_idx << ':' << y_data.max_idx
		          << "=" << y_data.idx_count() << std::endl;
	}

	MPI_Finalize();
	return 0;
}