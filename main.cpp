#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <utility>



typedef double coor_t;
typedef std::vector<coor_t> subarea_t;
typedef coor_t * coor_line_t;
typedef const coor_t (*func_t) (const coor_t x, const coor_t y);


const subarea_t operator-(const subarea_t & a, const subarea_t & b) {
	subarea_t tmp(a.size());
	for (int k = 0; k < a.size(); ++k) {
		tmp[k] = a[k] - b[k];
	}
	return tmp;
}

inline const coor_t F(const coor_t x, const coor_t y) {
	coor_t x_plus_y_2 = (x + y) * (x + y);
	return 4 * (1 - 2 * x_plus_y_2) * exp(1 - x_plus_y_2);
}


inline const coor_t phi(const coor_t x, const coor_t y) {
	return exp(1 - (x + y) * (x + y));
}

inline const coor_t solution(const coor_t x, const coor_t y) {
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

	inline bool is_global_max(const int global_idx) {
		return global_idx == N - 1;
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


struct func_data_t {
	coor_line_t up;
	coor_line_t down;
	coor_line_t left;
	coor_line_t right;
	subarea_t local;
	size_t x_idx_count;
	size_t y_idx_count;

	func_data_t(const size_t x_idx_count, const size_t y_idx_count):
			x_idx_count(x_idx_count),
			y_idx_count(y_idx_count),
			up(new coor_t[x_idx_count]),
			down(new coor_t[x_idx_count]),
			left(new coor_t[y_idx_count]),
			right(new coor_t[y_idx_count]),
			local(x_idx_count * y_idx_count)
	{}

	func_data_t & operator=(const func_data_t & another) {
		for (int i = 0; i < x_idx_count; ++i) {
			up[i] = another.up[i];
			down[i] = another.down[i];
		}
		for (int j = 0; j < y_idx_count; ++j) {
			left[j] = another.left[j];
			right[j] = another.right[j];
		}
		local = another.local;
		x_idx_count = another.x_idx_count;
		y_idx_count = another.y_idx_count;
		return *this;
	}

	bool operator==(const func_data_t & another) {
		for (int i = 0; i < x_idx_count; ++i) {
			if (up[i] != another.up[i] or down[i] != another.down[i]) {
				return false;
			}
		}
		for (int j = 0; j < y_idx_count; ++j) {
			if (left[j] != another.left[j] or right[j] != another.right[j]) {
				return false;
			}
		}
		return local == another.local
		       and x_idx_count == another.x_idx_count
		       and y_idx_count == another.y_idx_count;
	}

	~func_data_t() {
		delete[] up;
		delete[] down;
		delete[] left;
		delete[] right;
	}

};


std::ostream & operator<<(std::ostream & out, const func_data_t & func_data) {
//	int sz = func_data.local.size();
//	int sz = 1000 * 1000;
//	std::cout << sz << std::endl;
	for (int k = 0; k < func_data.local.size(); ++k) {
//	for (int k = 0; k < 1000 * 1000; ++k) {
//	for (int k = 0; k < sz; ++k) {
//		out << func_data.local[k] << ' ';
//		out << k << ' ';
	}
}


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
	enum func_tag_t {P_TAG, G_TAG, R_TAG};

	// names of service methods:
	// <type>_<order>_<place>_<belong>_<function>
	// <type> - is one of 'compute', 'calculate'
	// <order> - is one of 'init', 'next'
	// <place> - is one of 'internal', 'up', 'down', 'left', 'right', 'broad'
	// <belong> - is one of 'local', 'neighbor'
	// <function> - is one of 'p', 'r', 'g'



	// init value descriptions

	const coor_t compute_init_internal_p(const int i, const int j) {
//		return phi(x.coor(i), y.coor(j));
		return 0;
	}

	const coor_t compute_init_broad_p(const int i, const int j) {
		return phi(x.coor(i), y.coor(j));
	}

	const coor_t compute_init_internal_r(const int i, const int j) {
		return - delta_h(p, i, j) - F(x.coor(i), y.coor(j));
	}

	const coor_t compute_init_broad_r(const int i, const int j) {
		return 0;
	}

	inline const coor_t compute_init_local_g(const int i, const int j) {
		return get(r.local, i, j);
	}

	// init function for local function values

	inline const coor_t calculate_init_internal_p(const int i, const int j) {
		return put(p.local, i, j, compute_init_internal_p(i, j));
	}

	inline const coor_t calculate_init_broad_p(const int i, const int j) {
		return put(p.local, i, j, compute_init_broad_p(i, j));
	}

	inline const coor_t calculate_init_internal_r(const int i, const int j) {
		return put(r.local, i, j, compute_init_internal_r(i, j));
	}

	inline const coor_t calculate_init_broad_r(const int i, const int j) {
		return put(r.local, i, j, compute_init_broad_r(i, j));
	}

	inline const coor_t calculate_init_local_g(const int i, const int j) {
		return put(g.local, i, j, compute_init_local_g(i, j));
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

	inline const coor_t norm(const subarea_t & a) {
		return sqrt(scalar(a, a));
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
		return static_cast<size_t>(x.N * y.N);
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
			delete_statuses();
			p_responses = new MPI_Request[neighbor_counter];
			r_responses = new MPI_Request[neighbor_counter];
			g_responses = new MPI_Request[neighbor_counter];
		}
	}

	inline void delete_statuses() {
		delete[] p_responses;
		delete[] r_responses;
		delete[] g_responses;
	}






	void send_recv_to(func_tag_t func_tag, action_t send_act, action_t receive_act,
	                  coor_line_t sent_array, coor_line_t receive_array, MPI_Request &response,
	                  const int idx_count, const int target_rank
	) {
		MPI_Request request;
		// async send and receive to/from neighbors
		MPI_Isend(sent_array, idx_count, MPI_DOUBLE, target_rank,
		          get_tag(send_act, func_tag), MPI_COMM_WORLD, &request);
		MPI_Irecv(receive_array, idx_count, MPI_DOUBLE, target_rank,
		          get_tag(receive_act, func_tag), MPI_COMM_WORLD, &response);
	}

	void send_recv(sent_data_t & sent_data, func_data_t &func_data, func_tag_t func_tag, MPI_Request * responses) {
		int counter = 0;
		if (not is_global_up())
			send_recv_to(func_tag, SEND_UP, RECEIVE_UP, sent_data.sent_up, func_data.up,
			             responses[counter++], x.idx_count(), rank - x.side_processes_count());
		if (not is_global_down())
			send_recv_to(func_tag, SEND_DOWN, RECEIVE_DOWN, sent_data.sent_down, func_data.down,
			             responses[counter++], x.idx_count(), rank + x.side_processes_count());
		if (not is_global_left())
			send_recv_to(func_tag, SEND_LEFT, RECEIVE_LEFT, sent_data.sent_left, func_data.left,
			             responses[counter++], y.idx_count(), rank - 1);
		if (not is_global_right())
			send_recv_to(func_tag, SEND_RIGHT, RECEIVE_RIGHT, sent_data.sent_right, func_data.right,
			             responses[counter], y.idx_count(), rank + 1);
	}

	void wait_async_complete(MPI_Request * responses) {
		MPI_Status statuses[neighbor_counter];
		MPI_Waitall(neighbor_counter, responses, statuses);
	}

	void update_sent_vector(sent_data_t & sent_data, const int i, const int j, const coor_t value) {
		if (i == x.min_idx and not x.is_global_min(i)) {
			sent_data.sent_left[j] = value;
		}
		if (i == x.max_idx and not x.is_global_max(i)) {
			sent_data.sent_right[j] = value;
		}
		if (j == y.min_idx and not y.is_global_min(j)) {
			sent_data.sent_up[i] = value;
		}
		if (j == y.max_idx and not y.is_global_max(j)) {
			sent_data.sent_down[i] = value;
		}
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
	MPI_Request * p_responses;
	MPI_Request * r_responses;
	MPI_Request * g_responses;

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
			p_responses(NULL), r_responses(NULL), g_responses(NULL)
	{
		// compute neighbor counter
		if (not is_global_right())
			++neighbor_counter;
		if (not is_global_left())
			++neighbor_counter;
		if (not is_global_up())
			++neighbor_counter;
		if (not is_global_down())
			++neighbor_counter;
		// init 'p'
		for (int i = x.min_idx; i <= x.max_idx; ++i) {
			for (int j = y.min_idx; j <= y.max_idx; ++j) {
				// init local variables
				if (x.is_border(i) or x.is_border(j)) {
					calculate_init_broad_p(i, j);
				} else {
					calculate_init_internal_p(i, j);
				}
				calculate_init_local_g(i, j);
				// init neighbors variables
				if (j == y.min_idx and not is_global_up()) {
					p.up[i] = compute_init_internal_p(i, j);
				}
				if (j == y.max_idx and not is_global_down()) {
					p.down[i] = compute_init_internal_p(i, j);
				}
				if (i == x.min_idx and not is_global_left()) {
					p.left[j] = compute_init_internal_p(i, j);
				}
				if (i == x.max_idx and not is_global_right()) {
					p.right[j] = compute_init_internal_p(i, j);
				}
			}
		}
		// init 'r' and 'g'
		for (int i = x.min_idx; i <= x.max_idx; ++i) {
			for (int j = y.min_idx; j <= y.max_idx; ++j) {
				// init local variables
				if (x.is_border(i) or x.is_border(j)) {
					calculate_init_broad_r(i, j);
				} else {
					calculate_init_internal_r(i, j);
				}
				calculate_init_local_g(i, j);
				// init neighbors variables
				if (j == y.min_idx and not is_global_up()) {
					r.up[i] = compute_init_internal_r(i, j);
					g.up[i] = r.up[i];
				}
				if (j == y.max_idx and not is_global_down()) {
					r.down[i] = compute_init_internal_r(i, j);
					g.down[i] = r.down[i];
				}
				if (i == x.min_idx and not is_global_left()) {
					r.left[j] = compute_init_internal_r(i, j);
					g.left[j] = r.left[j];
				}
				if (i == x.max_idx and not is_global_right()) {
					r.right[j] = compute_init_internal_r(i, j);
					g.right[j] = r.right[j];
				}
			}
		}
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
		numerator = receive_data[0];
		denominator = receive_data[1];
		return numerator / denominator;
	}

	inline const coor_t compute_pij(const int i, const int j, const coor_t tau) {
		if (x.is_border(i) or y.is_border(j)) {
			return get(p.local, i, j);
		} else {
			return get(p.local, i, j) - tau * get(r.local, i, j);
		}
	}

	void calculate_new_p(const coor_t tau) {
		int i, j;
//		coor_t min_coor=0, max_coor=9e99;
		for (i = x.min_idx; i <= x.max_idx; ++i) {
			for (j = y.min_idx; j <= y.max_idx; ++j) {
				coor_t value = compute_pij(i, j, tau);
//				coor_t v = std::abs(value - phi(x.coor(i), y.coor(j)));
//				min_coor = v < min_coor ? v : min_coor;
//				max_coor = v < max_coor ? max_coor : v;
				put(new_p.local, i, j, value);
				update_sent_vector(sent_p, i, j, value);
			}
		}
//		std::cout << min_coor << "-" << max_coor << std::endl;

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
				coor_t value = compute_rij(i, j);
				put(new_r.local, i, j, value);
				update_sent_vector(sent_r, i, j, value);
			}
		}
	}

	const coor_t compute_alpha() {
		// compute local numerator and denominator for tau
		coor_t numerator = 0.0, denominator = 0.0;
		for (int i = x.min_idx; i <= x.max_idx; ++i) {
			for (int j = y.min_idx; j <= y.max_idx; ++j) {
				numerator += scalar_component( - delta_h(new_r, i, j), get(g.local, i, j), i, j);
				denominator += scalar_component( - delta_h(g, i, j), get(g.local, i, j), i, j);
			}
		}
		coor_t send_data[2], receive_data[2];
		send_data[0] = numerator;
		send_data[1] = denominator;
		MPI_Allreduce(send_data, receive_data, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		numerator = receive_data[0];
		denominator = receive_data[1];
		return numerator / denominator;
	}

	const coor_t compute_gij(const coor_t alpha, const int i, const int j) {
		return get(new_r.local, i, j) - alpha * get(g.local, i, j);
	}

	void calculate_new_g(const coor_t alpha) {
		for (int i = x.min_idx; i <= x.max_idx; ++i) {
			for (int j = y.min_idx; j <= y.max_idx; ++j) {
				coor_t value = compute_gij(alpha, i, j);
				put(new_g.local, i, j, value);
				update_sent_vector(sent_g, i, j, value);
			}
		}
	}

	std::pair<coor_t,coor_t> compute_difference_and_error() {
		// compute local difference and error for tau
		coor_t difference = 0.0, error = 0.0;
		for (int i = x.min_idx; i <= x.max_idx; ++i) {
			for (int j = y.min_idx; j <= y.max_idx; ++j) {
				coor_t buf1 = get(new_p.local, i, j) - get(p.local, i, j);
				difference += scalar_component(buf1, buf1, i, j);
				coor_t buf2 = get(new_p.local, i, j) - solution(x.coor(i), y.coor(j));
				error += scalar_component(buf2, buf2, i, j);
			}
		}
		coor_t send_data[2], receive_data[2];
		send_data[0] = difference;
		send_data[1] = error;
		MPI_Allreduce(send_data, receive_data, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		difference = sqrt(receive_data[0]);
		error = sqrt(receive_data[1]);
		return std::make_pair(difference, error);
	}

	std::pair<coor_t,coor_t> calculate_iteration() {
		init_statuses();

		std::cout << "compute tau" << std::endl;
		coor_t tau = compute_tau();

		std::cout << "compute p" << std::endl;
		calculate_new_p(tau);
//		std::cout << new_p.local.size() << std::endl;
		std::cout << new_p << std::endl;
		// notify about 'p'
		send_recv(sent_p, new_p, P_TAG, p_responses);
		// wait sync p
		wait_async_complete(p_responses);

		std::cout << "compute r" << std::endl;
		calculate_new_r();
		// notify about 'r'
		send_recv(sent_r, new_r, R_TAG, r_responses);
		// wait sync 'r'
		wait_async_complete(r_responses);

		std::cout << "compute alpha" << std::endl;
		coor_t alpha = compute_alpha();

		std::cout << "compute g" << std::endl;
		calculate_new_g(alpha);
		send_recv(sent_g, new_g, G_TAG, g_responses);
		wait_async_complete(g_responses);

		std::cout << "compute difference and error" << std::endl;
		return compute_difference_and_error();
	}


	void launch() {
		const int MAX_ITERATION = 1000;
		int iteration = 0;
		for (; iteration < MAX_ITERATION ;++iteration) {
			std::pair<coor_t, coor_t> result = calculate_iteration();
			coor_t difference = result.first;
			coor_t error = result.second;
			std::cout << iteration << ") " << difference << " " << error << std::endl;
			if (difference < eps)
				break;
			std::cout << "p == new_p is " << (p == new_p ? "true" : "false") << std::endl;
			p = new_p;
			r = new_r;
			g = new_g;
		}
	}

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
//	process_count = 12;

	std::cout << "process_count = " << process_count << std::endl;

	int a = static_cast<int>(sqrt(process_count));
//	for (; process_count % a > 0; --a)
//		;
	int b = process_count / a;
//	 N == a * b
//	 a - process number (X axes)
//	 b - process number (Y axes)
	// compute size on X axes
//	std::cout << "a = " << a << std::endl;
//	std::cout << "b = " << b << std::endl;
//	std::cout << "hear" << std::endl;
//	for (int i = 0; i < process_count; ++i) {
//		rank = i;
	std::cout << "rank " << rank << std::endl;
		std::pair<int, int> x_range = compute_subfield_size(rank, a, N, true, a);
		std::pair<int, int> y_range = compute_subfield_size(rank, b, N, false, a);
		OneDimensionData x_data = OneDimensionData(N, 0, 2, 3 / 2, x_range.first, x_range.second);
		OneDimensionData y_data = OneDimensionData(N, 0, 2, 3 / 2, y_range.first, y_range.second);

		std::cout << "init process" << std::endl;
		LocalProcess process(x_data, y_data, 0.0001, rank);
		std::cout << "launch" << std::endl;
		process.launch();
//
		std::cout << rank << " x " << x_data.min_idx << '-' << x_data.max_idx
		          << "=" << x_data.idx_count()
		          << " y " << y_data.min_idx << ':' << y_data.max_idx
		          << "=" << y_data.idx_count() << std::endl;
//	}

	MPI_Finalize();
	return 0;
}