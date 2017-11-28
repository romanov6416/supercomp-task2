#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <utility>




typedef double coor_t;
typedef std::vector<coor_t> subarea_t;
typedef coor_t * coor_line_t;
typedef const coor_t (*func_t) (const coor_t x, const coor_t y);

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


const coor_t delta_h(
		const coor_t c,
		const coor_t l, const coor_t r,
		const coor_t u, const coor_t d,
		const coor_t hi, const coor_t hi_1,
		const coor_t hj, const coor_t hj_1,
		const coor_t average_hi, const coor_t average_hj
) {
	//   u
	// l c r
	//   d
	coor_t first = ( (c-l)/hi_1 - (r-c)/hi ) / average_hi;
	coor_t second = ( (c-u)/hj_1 - (d-c)/hj ) / average_hj;
	return - (first + second);
}



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
//		return 0.5 * (coor(global_idx) + coor(global_idx + 1));
		return coor(global_idx + 1) - coor(global_idx);
	}

	inline const coor_t average_h(const int global_idx) {
		return 0.5 * (h(global_idx) + h(global_idx - 1));
	}

//	inline bool is_max(const coor_t coor) {
//		return std::abs(coor - B) < EPS;
//	}
//
//	inline bool is_min(const coor_t coor) {
//		return std::abs(coor - A) < EPS;
//	}

	inline bool is_global_max(const int global_idx) {
		return global_idx == N;
	}

	inline bool is_global_min(const int global_idx) {
		return global_idx == 0;
	}

	inline bool is_border(const int global_idx) {
		return is_global_min(global_idx) or is_global_max(global_idx);
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

	inline const size_t global_idx_count() {
		return static_cast<size_t>(N);
	}
};


enum action_t {
	SEND_UP, SEND_DOWN, SEND_LEFT, SEND_RIGHT,
	RECEIVE_UP=SEND_DOWN, RECEIVE_DOWN=SEND_UP,
	RECEIVE_LEFT=SEND_RIGHT, RECEIVE_RIGHT=SEND_LEFT
};

struct neighbor_info_t {
		action_t send_action;
		action_t receive_action;
};


struct func_data_t {
	coor_line_t up;
	coor_line_t down;
	coor_line_t left;
	coor_line_t right;
	subarea_t local;

	func_data_t(const size_t x_idx_count, const size_t y_idx_count):
			up(new coor_t[x_idx_count]),
			down(new coor_t[x_idx_count]),
			left(new coor_t[y_idx_count]),
			right(new coor_t[y_idx_count]),
			local(x_idx_count * y_idx_count)
	{}

	~func_data_t() {
		delete[] up;
		delete[] sent_up;
		delete[] down;
		delete[] sent_down;
		delete[] left;
		delete[] sent_left;
		delete[] right;
		delete[] sent_right;
	}

};


struct sent_data_t {
	coor_line_t sent_up;
	coor_line_t sent_down;
	coor_line_t sent_left;
	coor_line_t sent_right;

	sent_data_t(const size_t x_idx_count, const size_t y_idx_count):
			sent_up(new coor_t[x_idx_count]),
			sent_down(new coor_t[x_idx_count]),
			sent_left(new coor_t[y_idx_count]),
			sent_right(new coor_t[y_idx_count])
	{}

	~sent_data_t() {
		delete[] sent_up;
		delete[] sent_down;
		delete[] sent_left;
		delete[] sent_right;
	}


};


class LocalProcess {
private:
	enum func_tag_t {TAU};

	// names of service methods:
	// <type>_<order>_<place>_<belong>_<function>
	// <type> - is one of 'compute', 'calculate'
	// <order> - is one of 'init', 'next'
	// <place> - is one of 'internal', 'up', 'down', 'left', 'right', 'broad'
	// <belong> - is one of 'local', 'neighbor'
	// <function> - is one of 'p', 'r', 'g'

	// init value descriptions

	inline const coor_t compute_init_internal_p(const int i, const int j) {
		return phi(x.coor(i), y.coor(j));
	}

	inline const coor_t compute_init_internal_r(const int i, const int j) {
		return - delta_h(phi, i, j) - F(x.coor(i), y.coor(j));
	}

	inline const coor_t compute_init_local_g(const int i, const int j) {
		return get(r.local, i, j);
	}

	// init function of local

	inline void calculate_init_internal_local(const int i, const int j) {
		calculate_init_internal_local_p(i, j);
		calculate_init_internal_local_r(i, j);
		calculate_init_local_g(i, j);
	}

	inline const coor_t calculate_init_internal_local_p(const int i, const int j) {
		return put(p.local, i, j, compute_init_internal_p(i, j));
	}

	inline const coor_t calculate_init_internal_local_r(const int i, const int j) {
		return put(r.local, i, j, compute_init_internal_r(i, j));
	}

	inline const coor_t calculate_init_local_g(const int i, const int j) {
		return put(g.local, i, j, compute_init_local_g(i, j));
	}




	inline void calculate_init_broad_local(const int i, const int j) {
		calculate_init_broad_local_p(i, j);
		init_broad_local_r(i, j);
		calculate_init_local_g(i, j);
	}

	inline const coor_t calculate_init_broad_local_p(const int i, const int j) {
		return put(p.local, i, j, 0);  // can be any value
	}

	inline const coor_t init_broad_local_r(const int i, const int j) {
		return put(r.local, i, j, 0);
	}

	// init neighbors

	inline void calculate_init_neighbor_down(const int i, const int j) {
		p.down[i] = compute_init_internal_p(i, j + 1);
		r.down[i] = compute_init_internal_r(i, j + 1);
		g.down[i] = r.down[i];
	}

	inline void calculate_init_neighbor_up(const int i, const int j) {
		p.up[i] = compute_init_internal_p(i, j - 1);
		r.up[i] = compute_init_internal_r(i, j - 1);
		g.up[i] = r.up[i];
	}

	inline void calculate_init_neighbor_left(const int i, const int j) {
		p.left[j] = compute_init_internal_p(i - 1, j);
		r.left[j] = compute_init_internal_r(i - 1, j);
		g.left[j] = r.left[j];
	}

	inline void calculate_init_neighbor_right(const int i, const int j) {
		p.right[j] = compute_init_internal_p(i + 1, j);
		r.right[j] = compute_init_internal_r(i + 1, j);
		g.right[j] = r.right[j];
	}




	inline const int get_tag(action_t act, func_tag_t v) {
		return act + (v << 2);
	}

	inline const coor_t delta_h(const func_data_t & a, const int i, const int j) {
		return ::delta_h(
				get(a.local, i, j), // center
				i-1 < x.min_idx ? a.left[j] : get(a.local, i-1, j), // left
				i+1 > x.max_idx ? a.right[j] : get(a.local, i+1, j), // right
				j-1 < y.min_idx ? a.up[i] : get(a.local, i, j-1), // up
				j+1 > y.max_idx ? a.down[i] : get(a.local, i, j+1), // down
				x.h(i), x.h(i - 1), y.h(j), y.h(j - 1),
				x.average_h(i), y.average_h(j)
		);
	}

	inline const coor_t delta_h(func_t func, const int i, const int j) {
		return ::delta_h(
				func(x.coor(i), y.coor(j)),
				func(x.coor(i-1), y.coor(j)), func(x.coor(i+1), y.coor(j)),
				func(x.coor(i), y.coor(j-1)), func(x.coor(i), y.coor(j+1)),
				x.h(i), x.h(i - 1), y.h(j), y.h(j - 1),
				x.average_h(i), y.average_h(j)
		);
	}

	inline const coor_t scalar_component(const coor_t aij, const coor_t bij, const int i, const int j) {
		return aij * bij * x.average_h(i) * y.average_h(j);
	}

	inline const coor_t scalar(const subarea_t & a, const subarea_t & b) {
		coor_t s = 0.0;
		for (int i = x.min_idx; i <= x.max_idx; ++i) {
			for (int j = y.min_idx; j <= y.max_idx; ++j) {
				s += scalar_component(get(a, i, j), get(b, i, j), i, j);
			}
		}
	}

	inline bool is_global_left() {
		return x.is_global_min(x.min_idx);
	}

	inline bool is_global_right() {
		return x.is_global_max(x.max_idx);
	}

	inline bool is_global_up() {
		return y.is_global_min(y.min_idx);
	}

	inline bool is_global_down() {
		return y.is_global_max(y.max_idx);
	}










	inline size_t size() {
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


	inline void init_statuses() {
		// init request statuses
		if (neighbor_counter) {
			p_statuses = new MPI_Request[neighbor_counter];
			r_statuses = new MPI_Request[neighbor_counter];
			g_statuses = new MPI_Request[neighbor_counter];
		}
	}

	inline void delete_statuses() {
		delete[] p_statuses;
		delete[] r_statuses;
		delete[] g_statuses;
	}


public:
	OneDimensionData x;
	OneDimensionData y;
	const coor_t eps;
	int rank;
	func_data_t p;
	func_data_t new_p;
	sent_data_t sent_p;
	func_data_t r;
	func_data_t new_r;
	sent_data_t sent_r;
	func_data_t g;
	func_data_t new_g;
	sent_data_t sent_g;
	int neighbor_counter;
	MPI_Request * p_statuses;
	MPI_Request * r_statuses;
	MPI_Request * g_statuses;

	LocalProcess(const OneDimensionData & x_data, const OneDimensionData & y_data, const coor_t eps, const int rank):
			x(x_data), y(y_data), rank(rank), eps(eps),
			p(x.global_idx_count(), y.global_idx_count()),
			new_p(x.global_idx_count(), y.global_idx_count()),
			sent_p(x.global_idx_count(), y.global_idx_count()),
			r(x.global_idx_count(), y.global_idx_count()),
			new_r(x.global_idx_count(), y.global_idx_count()),
			sent_r(x.global_idx_count(), y.global_idx_count()),
			g(x.global_idx_count(), y.global_idx_count()),
			new_g(x.global_idx_count(), y.global_idx_count()),
			sent_g(x.global_idx_count(), y.global_idx_count()),
			neighbor_counter(0),
			p_statuses(NULL), r_statuses(NULL), g_statuses(NULL)
	{
		// compute neighbor counter
		if (is_global_right())
			++neighbor_counter;
		if (is_global_left())
			++neighbor_counter;
		if (is_global_up())
			++neighbor_counter;
		if (is_global_down())
			++neighbor_counter;

		// init for internal points
		int i, j;
		for (i = x.min_idx + 1; i <= x.max_idx - 1; ++i) {
			for (j = y.min_idx + 1; j <= y.max_idx - 1; ++j) {
				calculate_init_internal_local(i, j);
			}
		}
		// init left process broad
		i = x.min_idx;
		if (x.is_global_min(i)) {
			// the process is the most left
			for (j = y.min_idx; j <= y.max_idx; ++j) {
				calculate_init_broad_local(i, j);
			}
		} else {
			// init internal right
			for (j = y.min_idx + 1; j <= y.max_idx - 1; ++j) {
				calculate_init_neighbor_left(i, j);
				calculate_init_internal_local(i, j);
			}
			// init left up point
			j = y.min_idx;
			if (y.is_global_min(j)) {
				calculate_init_broad_local(i, j);
			} else {
				calculate_init_neighbor_left(i, j);
				calculate_init_internal_local(i, j);
			}
			// init left down point
			j = y.max_idx;
			if (y.is_global_max(j)) {
				calculate_init_broad_local(i, j);
			} else {
				calculate_init_neighbor_left(i, j);
				calculate_init_internal_local(i, j);
			}
		}
		// init right process broad
		i = x.max_idx;
		if (x.is_global_max(i)) {
			// the process is the most right
			for (j = y.min_idx; j <= y.max_idx; ++j) {
				calculate_init_broad_local(i, j);
			}
		} else {
			// init internal right
			for (j = y.min_idx + 1; j <= y.max_idx - 1; ++j) {
				calculate_init_neighbor_right(i, j);
				calculate_init_internal_local(i, j);
			}
			// init right up point
			j = y.min_idx;
			if (y.is_global_min(j)) {
				calculate_init_broad_local(i, j);
			} else {
				calculate_init_neighbor_right(i, j);
				calculate_init_internal_local(i, j);
			}
			// init right down point
			j = y.max_idx;
			if (y.is_global_max(j)) {
				calculate_init_broad_local(i, j);
			} else {
				calculate_init_neighbor_right(i, j);
				calculate_init_internal_local(i, j);
			}
		}
		// init up process broad
		j = y.min_idx;
		if (y.is_global_min(i)) {
			// the process is the most up
			for (i = x.min_idx; i <= y.max_idx; ++i) {
				calculate_init_broad_local(i, j);
			}
		} else {
			// init internal up points
			for (i = x.min_idx + 1; i <= y.max_idx - 1; ++i) {
				calculate_init_neighbor_up(i, j);
				calculate_init_internal_local(i, j);
			}
			// init left up point
			i = x.min_idx;
			if (x.is_global_min(i)) {
				calculate_init_broad_local(i, j);
			} else {
				calculate_init_neighbor_up(i, j);
				calculate_init_internal_local(i, j);
			}
			// init right up point
			i = x.max_idx;
			if (x.is_global_max(i)) {
				calculate_init_broad_local(i, j);
			} else {
				calculate_init_neighbor_up(i, j);
				calculate_init_internal_local(i, j);
			}
		}
		// init down process broad
		j = y.max_idx;
		if (y.is_global_max(i)) {
			// the process is the most down
			for (i = x.min_idx; i <= y.max_idx; ++i) {
				calculate_init_broad_local(i, j);
			}
		} else {
			// init internal down points
			for (i = x.min_idx + 1; i <= y.max_idx - 1; ++i) {
				calculate_init_neighbor_down(i, j);
				calculate_init_internal_local(i, j);
			}
			// init left down point
			i = x.min_idx;
			if (x.is_global_min(i)) {
				calculate_init_broad_local(i, j);
			} else {
				calculate_init_neighbor_down(i, j);
				calculate_init_internal_local(i, j);
			}
			// init right down point
			i = x.max_idx;
			if (x.is_global_max(i)) {
				calculate_init_broad_local(i, j);
			} else {
				calculate_init_neighbor_down(i, j);
				calculate_init_internal_local(i, j);
			}
		}
	}

	void send_recv_async_neighbors(
			func_data_t & func_data, func_tag_t func_tag,
			MPI_Request & up_response, MPI_Request & down_response,
			MPI_Request & left_response, MPI_Request & right_response
	) {
		// copy local vector 'r' for sending neighbors
		coor_t sent_up[x.N], sent_down[x.N], sent_left[y.N], sent_right[y.N];
		for (int i = x.min_idx; i <= x.max_idx; ++i) {
			if (x.is_border(i))
				continue;
			if (not y.is_border(y.min_idx))
				put(sent_up, i, y.min_idx, get(func_data.local, i, y.min_idx));
			if (not y.is_border(y.max_idx))
				put(sent_down, i, y.max_idx, get(func_data.local, i, y.max_idx));
		}
		for (int j = y.min_idx; j <= y.max_idx; ++j) {
			if (y.is_border(j))
				continue;
			if (not x.is_border(x.min_idx))
				put(sent_left, x.min_idx, j, get(func_data.local, x.min_idx, j));
			if (not x.is_border(x.max_idx))
				put(sent_right, x.max_idx, j, get(func_data.local, x.max_idx, j));
		}
		// async send and receive local vector 'r' to/from neighbors
//		coor_t received_up[x.N], received_down[x.N], received_left[y.N], received_right[y.N];
		MPI_Request up_request, down_request, left_request, right_request;
//		MPI_Request  * up_resp_ptr = respo, down_response, left_response, right_response;
		if (not y.is_border(y.min_idx)) {
			MPI_Isend(sent_up, x.N, MPI_DOUBLE, rank - x.side_processes_count(),
			          get_tag(SEND_UP, func_tag), MPI_COMM_WORLD, &up_request);
			MPI_Irecv(func_data.up, x.N, MPI_DOUBLE, rank - x.side_processes_count(),
			          get_tag(RECEIVE_UP, func_tag), MPI_COMM_WORLD, &up_response);
		}
		if (not y.is_border(y.max_idx)) {
			MPI_Isend(sent_down, x.N, MPI_DOUBLE, rank + x.side_processes_count(),
			          get_tag(SEND_DOWN, func_tag), MPI_COMM_WORLD, &down_request);
			MPI_Irecv(func_data.down, x.N, MPI_DOUBLE, rank - x.side_processes_count(),
			          get_tag(RECEIVE_DOWN, func_tag), MPI_COMM_WORLD, &down_response);
		}
		if (not x.is_border(x.min_idx)) {
			MPI_Isend(sent_left, y.N, MPI_DOUBLE, rank - y.side_processes_count(),
			          get_tag(SEND_LEFT, func_tag), MPI_COMM_WORLD, &left_request);
			MPI_Irecv(func_data.left, y.N, MPI_DOUBLE, rank - y.side_processes_count(),
			          get_tag(RECEIVE_LEFT, func_tag), MPI_COMM_WORLD, &left_response);
		}
		if (not x.is_border(x.max_idx)) {
			MPI_Isend(sent_right, y.N, MPI_DOUBLE, rank + y.side_processes_count(),
			          get_tag(SEND_RIGHT, func_tag), MPI_COMM_WORLD, &right_request);
			MPI_Irecv(func_data.right, y.N, MPI_DOUBLE, rank - y.side_processes_count(),
			          get_tag(RECEIVE_RIGHT, func_tag), MPI_COMM_WORLD, &right_response);
		}
	}

	void send_recv_async_up(func_data_t & func_data, func_tag_t func_tag, MPI_Request & response) {
		// copy to sent_data
		coor_t sent_data[x.N];
		for (int i = x.min_idx; i <= x.max_idx; ++i) {
			sent_data[i] = get(func_data.local, i, y.min_idx);
		}
		MPI_Request request;
		// async send and receive to/from neighbors
		MPI_Isend(sent_data, x.N, MPI_DOUBLE, rank - x.side_processes_count(),
		          get_tag(SEND_UP, func_tag), MPI_COMM_WORLD, &request);
		MPI_Irecv(func_data.up, x.N, MPI_DOUBLE, rank - x.side_processes_count(),
		          get_tag(RECEIVE_UP, func_tag), MPI_COMM_WORLD, &response);
	}

	void wait_async_complete(MPI_Request & up_response, MPI_Request & down_response,
	                         MPI_Request & left_response, MPI_Request & right_response) {
		int counter = 0;
		if (not is_global_up()) {
			++counter;
		}
		if (not is_global_down()) {
			++counter;
		}
		if (not is_global_left()) {
			++counter;
		}
		if (not is_global_right()) {
			++counter;
		}
		MPI_Status statuses[counter];
		MPI_Request responses[counter];
		MPI_Waitall(counter, responses, statuses);
	}

	const coor_t compute_tau() {
		// compute local numerator and denominator for tau
		coor_t numerator = 0.0, denominator = 0.0;
		for (int i = x.min_idx; i <= x.max_idx; ++i) {
			for (int j = y.min_idx; j <= y.max_idx; ++j) {
				numerator += scalar_component(get(r.local, i, j), get(r.local, i, j), i, j);
				denominator -= scalar_component(delta_h(r, i, j), get(r.local, i, j), i, j);
			}
		}
		coor_t send_data[2], receive_data[2];
		send_data[0] = numerator;
		send_data[1] = denominator;
		MPI_Allreduce(send_data, receive_data, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		return numerator / denominator;
	}

	inline const coor_t compute_pij(const int i, const int j, const coor_t tau) {
		return get(p.local, i, j) - tau * get(r.local, i, j);
	}

	inline void calculate_new_pij(const int i, const int j, const coor_t tau) {
		put(new_p.local, i, j, compute_pij(i, j, tau));
	}

	void calculate_new_p(
			MPI_Request & up_response, MPI_Request & down_response,
			MPI_Request & left_response, MPI_Request & right_response
	) {
		coor_t tau = compute_tau();
		int i, j;
		for (i = x.min_idx + 1; i <= x.max_idx - 1; ++i) {
			for (j = y.min_idx + 1; j <= y.max_idx - 1; ++j) {
				calculate_new_pij(i, j, tau);
			}
		}
		i = x.min_idx;
		if (x.is_global_min(i)) {

		} else {
			for (j = y.min_idx - 1; j <= y.max_idx - 1; ++j) {
				put(new_p.local, i, j, compute_pij(i, j, tau));
			}
		}
	}

	const coor_t compute_rij(const int i, const int j) {
		if (x.is_border(i) or y.is_border(j))
			return 0;
		else
			return - delta_h(new_p, i, j) - F(x.coor(i), y.coor(j));
	}

	void calculate_new_r() {
		for (int i = x.min_idx; i <= x.max_idx; ++i) {
			for (int j = y.min_idx; j <= y.max_idx; ++j) {
				compute_rij(i, j);
			}
		}
	}

//	void calculate_iteration() {
//		calculate_new_p();
//
//		calculate_new_r();
//	}

	~LocalProcess() {
		delete_statuses();
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