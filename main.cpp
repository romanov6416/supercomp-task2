#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <utility>



typedef double coor_t;
typedef std::vector<coor_t> subarea_t;
typedef coor_t * coor_line_t;
typedef coor_t (*func_t) (const coor_t x, const coor_t y);

enum func_tag_t {P_TAG, G_TAG, R_TAG, NULL_TAG};

enum action_t {
	SEND_UP, SEND_DOWN, SEND_LEFT, SEND_RIGHT,
	RECEIVE_UP=SEND_DOWN, RECEIVE_DOWN=SEND_UP,
	RECEIVE_LEFT=SEND_RIGHT, RECEIVE_RIGHT=SEND_LEFT
};


inline int get_tag(action_t act, func_tag_t v) {
	return act + (v << 2);
}




struct func_data_t {
	std::vector< std::vector<coor_t> > data;
	func_tag_t tag;

	func_data_t(func_tag_t tag): tag(tag) {}

	inline void resize(const size_t x_size, const size_t y_size) {
		data.resize(x_size);
		for (size_t i = 0; i < data.size(); ++i)
			data[i].resize(y_size);
	}

	inline size_t size_x() {
		return data.size();
	}

	inline size_t size_y() {
		return data[0].size();
	}

	inline const coor_t & operator()(const size_t i, const size_t j) const {
		return data[i][j];
	}

	coor_t & operator()(const size_t local_i, const size_t local_j) {
		return data[local_i][local_j];
	}

	bool operator==(const func_data_t & another) {
		return data == another.data;
	}

	func_data_t operator-(const func_data_t & other) const {
		func_data_t tmp(tag);
		tmp.data.resize(data.size());
		for (size_t i = 0; i < data.size(); ++i) {
			tmp.data[i].resize(data[i].size());
			for (size_t j = 0; j < data[i].size(); ++j) {
				tmp.data[i][j] = data[i][j] - other.data[i][j];
			}
		}
		return tmp;
	}
};

std::ostream & operator<<(std::ostream & out, func_data_t & func_data) {
	for (size_t i = 0; i < func_data.size_x(); ++i) {
		for (size_t j = 0; j < func_data.size_y(); ++j) {
			out << func_data(i,j) << ' ';
		}
	}
	return out;
}


class ISendReceive {
public:
	virtual void send_recv(func_data_t & func_data) = 0;
	virtual void wait(func_data_t & func_data) = 0;
	virtual ~ISendReceive() {};
};


class BaseSendReceive: public ISendReceive {
public:
	action_t send_dir;
	action_t receive_dir;
	MPI_Request request;
	MPI_Request response;
	const int target_rank;

	BaseSendReceive(action_t send_direction, action_t receive_direction, const int target_rank):
		send_dir(send_direction), receive_dir(receive_direction),
		request(), response(), target_rank(target_rank)
	{}


	void mpi_send_mpi_recv(func_tag_t func_tag, const size_t array_size,
	                       coor_t * sent_array, coor_t * receive_array) {
		// async send and receive to/from neighbors
		MPI_Isend(sent_array, static_cast<int>(array_size), MPI_DOUBLE, target_rank,
		          get_tag(send_dir, func_tag), MPI_COMM_WORLD, &request);
		MPI_Irecv(receive_array, static_cast<int>(array_size), MPI_DOUBLE, target_rank,
		          get_tag(receive_dir, func_tag), MPI_COMM_WORLD, &response);
	}

};


class SendReceiveEmpty: public ISendReceive {
public:
	virtual void send_recv(func_data_t & ) {}
	virtual void wait(func_data_t &) {}
};


class SendReceiveUp: public BaseSendReceive {
	std::vector<coor_t> sent_data;
	std::vector<coor_t> receive_data;

public:
	SendReceiveUp(const size_t size, const int target_rank):
			BaseSendReceive(SEND_UP, RECEIVE_UP, target_rank),
			sent_data(size), receive_data(size)
	{}

	virtual void send_recv(func_data_t & func_data) {
		for (size_t i = 0; i < sent_data.size(); ++i)
			sent_data[i] = func_data(i,1);
		mpi_send_mpi_recv(func_data.tag, func_data.size_x(), &sent_data[0], &receive_data[0]);
	}

	virtual void wait(func_data_t & func_data) {
		MPI_Status status;
		MPI_Wait(&request, &status);
		for (size_t i = 0; i < receive_data.size(); ++i)
			func_data(i,0) = receive_data[i];
	}
};


class SendReceiveDown: public BaseSendReceive {
	std::vector<coor_t> sent_data;
	std::vector<coor_t> receive_data;

public:
	SendReceiveDown(const size_t size, const int target_rank):
			BaseSendReceive(SEND_DOWN, RECEIVE_DOWN, target_rank),
			sent_data(size), receive_data(size)
	{}

	virtual void send_recv(func_data_t & func_data) {
		for (size_t j = 0; j < sent_data.size(); ++j)
			sent_data[j] = func_data(j, func_data.size_y() - 2);
		mpi_send_mpi_recv(func_data.tag, func_data.size_x(), &sent_data[0], &receive_data[0]);
	}

	virtual void wait(func_data_t & func_data) {
		MPI_Status status;
		MPI_Wait(&request, &status);
		for (size_t i = 0; i < receive_data.size(); ++i)
			func_data(i, func_data.size_y() - 1) = receive_data[i];
	}
};


class SendReceiveLeft: public BaseSendReceive {

public:
	SendReceiveLeft(const int target_rank):
			BaseSendReceive(SEND_LEFT, RECEIVE_LEFT, target_rank)
	{}

	virtual void send_recv(func_data_t & func_data) {
		mpi_send_mpi_recv(func_data.tag, func_data.size_y(),
		                  &func_data(1,0), &func_data(0,0));
	}

	virtual void wait(func_data_t &) {
		MPI_Status status;
		MPI_Wait(&request, &status);
	}
};


class SendReceiveRight: public BaseSendReceive {

public:
	SendReceiveRight(const int target_rank):
			BaseSendReceive(SEND_RIGHT, RECEIVE_RIGHT, target_rank)
	{}

	virtual void send_recv(func_data_t & func_data) {
		mpi_send_mpi_recv(func_data.tag, func_data.size_y(),
		                  &func_data(func_data.size_x() - 2,0),
		                  &func_data(func_data.size_x() - 1,0));
	}

	virtual void wait(func_data_t &) {
		MPI_Status status;
		MPI_Wait(&request, &status);
	}
};




const subarea_t operator-(const subarea_t & a, const subarea_t & b) {
	subarea_t tmp(a.size());
	for (size_t k = 0; k < a.size(); ++k) {
		tmp[k] = a[k] - b[k];
	}
	return tmp;
}

inline coor_t F(const coor_t x, const coor_t y) {
	coor_t x_plus_y_2 = (x + y) * (x + y);
	return 4 * (1 - 2 * x_plus_y_2) * exp(1 - x_plus_y_2);
}

inline coor_t phi(const coor_t x, const coor_t y) {
	return exp(1 - (x + y) * (x + y));
//	return 1 + sin(x*y);
}

inline coor_t solution(const coor_t x, const coor_t y) {
	return phi(x, y);
}


class OneDimensionData {
public:
	const int N;  // number of points
	const coor_t A;  // begin of segment
	const coor_t B;  // end of segment
	const coor_t q;  // eventually ratio
	const size_t min_idx;
	const size_t max_idx;
	std::vector<coor_t> line;

	OneDimensionData(const int N, const coor_t A, const coor_t B, const coor_t q,
	                 const size_t min_local_idx, const size_t max_local_idx) :
			N(N), A(A), B(B), q(q), min_idx(min_local_idx), max_idx(max_local_idx), line(N)
	{
		for (int i = 0; i < N; ++i) {
			line[i] = _coor(static_cast<size_t>(i));
		}
	}

	inline coor_t f(const coor_t t) {
		return (pow(1 + t, q) - 1.0) / (pow(2, q) - 1);
	}

	inline coor_t coor(const size_t local_idx) {
		return line[global(local_idx)];
	}

	inline coor_t _coor(const size_t global_idx) {
		coor_t t = static_cast<coor_t>(global_idx) / N;
		return B * f(t) + A * (1 - f(t));
	}

	inline coor_t h(const size_t local_idx) {
		return coor(local_idx + 1) - coor(local_idx);
	}

	inline coor_t average_h(const size_t local_idx) {
		return 0.5 * (h(local_idx) + h(local_idx - 1));
	}

	inline bool is_global_max(const size_t local_idx) {
		return global(local_idx) == static_cast< size_t >(N - 1);
	}

	inline bool is_global_min(const size_t local_idx) {
		return global(local_idx) == 0;
	}

	inline bool is_border(const size_t local_idx) {
		return is_global_min(local_idx) or is_global_max(local_idx);
	}

	inline size_t local(const int global_idx) {
		return global_idx - min_idx;
	}

	inline size_t idx_count() {
		return max_idx - min_idx + 1;
	}

	inline int side_processes_count() {
		return static_cast<int>(round(static_cast<double>(N) / idx_count()));
	}

	inline size_t global_idx_count() {
		return static_cast<size_t>(N);
	}

	template <typename T>
	inline T global(const T local_idx) {
		return static_cast<T>(min_idx) + local_idx;
	}

	coor_t operator[](const size_t i) {
		return coor(static_cast<int>(i));
	}
};



class iteration_data_t {
public:
	func_data_t p, r, g;

	iteration_data_t(): p(P_TAG), r(R_TAG), g(G_TAG) {}

	bool operator==(const iteration_data_t & other) {
		return p == other.p and r == other.r and g == other.g;
	}

	void resize(const size_t x_size, const size_t y_size) {
		p.resize(x_size, y_size);
		r.resize(x_size, y_size);
		g.resize(x_size, y_size);
	}

};


class LocalProcess {
private:

	inline coor_t delta_h(const func_data_t & a, const size_t i, const size_t j) {
		coor_t l = (a(i,j) - a(i-1,j)) / x.h(i-1)   -   (a(i+1,j) - a(i,j)) / x.h(i);// / average_hi;
		coor_t r = (a(i,j) - a(i,j-1)) / y.h(j-1)   -   (a(i,j+1) - a(i,j)) / y.h(j);// ) / average_hj;
		return - (l / x.average_h(i) + r / y.average_h(j));
	}

	inline coor_t scalar_component(const coor_t aij, const coor_t bij, const size_t i, const size_t j) {
		return aij * bij * x.average_h(i) * y.average_h(j);
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

	inline bool is_border(const size_t i, const size_t j) {
		return x.is_border(i) or y.is_border(j);
	}

	void send_recv(func_data_t & func_data) {
		up_neighbor->send_recv(func_data);
		down_neighbor->send_recv(func_data);
		left_neighbor->send_recv(func_data);
		right_neighbor->send_recv(func_data);
	}

	void wait(func_data_t & func_data) {
		up_neighbor->wait(func_data);
		down_neighbor->wait(func_data);
		left_neighbor->wait(func_data);
		right_neighbor->wait(func_data);
	}



public:
	OneDimensionData x;
	OneDimensionData y;
	const coor_t eps;
	int rank;

	iteration_data_t cur;
	iteration_data_t next;
	size_t x_size;
	size_t y_size;
	ISendReceive * up_neighbor;
	ISendReceive * down_neighbor;
	ISendReceive * left_neighbor;
	ISendReceive * right_neighbor;

//	func_data_t phi;
//	func_data_t F;

	LocalProcess(const OneDimensionData & x_data, const OneDimensionData & y_data, const coor_t eps, const int rank):
			x(x_data), y(y_data), eps(eps), rank(rank),
			x_size(static_cast<size_t>(x.N)), y_size(static_cast<size_t>(y.N)),
			up_neighbor(NULL), down_neighbor(NULL),
			left_neighbor(NULL), right_neighbor(NULL)
//			phi(NULL_TAG), F(NULL_TAG)
	{
		if (not is_global_up())
			++x_size;
		if (not is_global_down())
			++x_size;
		if (not is_global_left())
			++y_size;
		if (not is_global_right())
			++y_size;

		if (is_global_up()) {
			up_neighbor = new SendReceiveEmpty();
		} else {
			up_neighbor = new SendReceiveUp(x_size, rank - x.side_processes_count());
		}
		if (is_global_down()) {
			down_neighbor = new SendReceiveEmpty();
		} else {
			down_neighbor = new SendReceiveDown(x_size, rank + x.side_processes_count());
		}
		if (is_global_left()) {
			left_neighbor = new SendReceiveEmpty();
		} else {
			left_neighbor = new SendReceiveLeft(rank - 1);
		}
		if (is_global_right()) {
			right_neighbor = new SendReceiveEmpty();
		} else {
			right_neighbor = new SendReceiveRight(rank + 1);
		}

		cur.resize(x_size, y_size);
		next.resize(x_size, y_size);

		if (rank == 0) std::cout << "init 'p'" << std::endl;
		for (size_t i = 0; i < x_size; ++i) {
			for (size_t j = 0; j < y_size; ++j) {
//				cur.p(i,j) = is_border(i,j) ? phi(x[i],y[j]) : 0;
				cur.p(i,j) = is_border(i,j) ? phi(x[i],y[j]) : phi(x[i],y[j]);
//				cur.p(i,j) = is_border(i,j) ? phi(i,j) : 0;
			}
		}

		// init 'r' and 'g'
		if (rank == 0) std::cout << "init 'r' and 'g'" << std::endl;
		for (size_t i = 0; i < x_size; ++i) {
			for (size_t j = 0; j < y_size; ++j) {
				cur.r(i,j) = is_border(i,j) ? 0 : - delta_h(cur.p, i, j) - F(x[i], y[j]);
//				cur.r(i,j) = is_border(i,j) ? 0 : - delta_h(cur.p, i, j) - F(i,j);
				cur.g(i,j) = cur.r(i,j);
			}
		}
	}

	inline coor_t compute_tau() {
		// compute local numerator and denominator for tau
		coor_t numerator = 0.0, denominator = 0.0;
		for (size_t i = 1; i < x_size - 1; ++i) {
			for (size_t j = 1; j < y_size - 1; ++j) {
				numerator += scalar_component(cur.r(i,j), cur.r(i,j), i, j);
				denominator -= scalar_component(delta_h(cur.r,i,j) , cur.r(i,j), i, j);
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

	void calculate_new_p(const coor_t tau) {
		for (size_t i = 1; i < x_size - 1; ++i) {
			for (size_t j = 1; j < y_size - 1; ++j) {
				next.p(i,j) = cur.p(i,j) - tau * cur.r(i,j);
			}
		}
	}

	inline void calculate_new_r() {
		for (size_t i = 1; i < x_size - 1; ++i) {
			for (size_t j = 1; j < y_size - 1; ++j) {
				next.r(i,j) = - delta_h(next.p, i, j) - F(x[i], y[j]);
//				next.r(i,j) = - delta_h(next.p, i, j) - F(i,j);
			}
		}
	}

	inline coor_t compute_alpha() {
		// compute local numerator and denominator for tau
//		double start = MPI_Wtime();
		coor_t numerator = 0.0, denominator = 0.0;
		for (size_t i = 1; i < x_size - 1; ++i) {
			for (size_t j = 1; j < y_size - 1; ++j) {
				numerator -= scalar_component(delta_h(next.r, i, j), cur.g(i, j), i, j);
				denominator -= scalar_component(delta_h(cur.g, i, j), cur.g(i, j), i, j);
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

	void calculate_new_g(const coor_t alpha) {
		for (size_t i = 0; i < x_size; ++i) {
			for (size_t j = 0; j < y_size; ++j) {
				next.g(i,j) = next.r(i,j) - alpha * cur.g(i,j);
			}
		}
	}

	std::pair<coor_t,coor_t> compute_difference_and_error() {
		// compute local difference and error for tau
		coor_t difference = 0.0, error = 0.0;
		for (size_t i = 1; i < x_size - 1; ++i) {
			for (size_t j = 1; j < y_size - 1; ++j) {
				coor_t buf = next.p(i,j) - cur.p(i,j);
				difference += scalar_component(buf, buf, i, j);
				buf = next.p(i,j) - solution(x[i], y[j]);
				error += scalar_component(buf, buf, i, j);
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
//		double start, end;

//		start = MPI_Wtime();
		coor_t tau = compute_tau();
//		end = MPI_Wtime();
//		if(rank == 0) std::cout << "computed tau (" << end - start << ")" << std::endl;
//		start = MPI_Wtime();
		calculate_new_p(tau);
		send_recv(next.p);
		wait(next.p);
//		end = MPI_Wtime();
//		if (rank == 0) std::cout << "computed p (" << end - start << ")" << std::endl;


//		start = MPI_Wtime();
		calculate_new_r();
		send_recv(next.r);
		wait(next.r);
//		end = MPI_Wtime();
//		if (rank == 0) std::cout << "computed r (" << end - start  << ")" << std::endl;


//		start = MPI_Wtime();
		coor_t alpha = compute_alpha();
//		end = MPI_Wtime();
//		if (rank == 0) std::cout << "computed alpha (" << end - start << ")" << std::endl;

//		start = MPI_Wtime();
		calculate_new_g(alpha);
		send_recv(next.g);
		wait(next.g);
//		end = MPI_Wtime();
//		if (rank == 0) std::cout << "computed g (" << end - start << ")" << std::endl;

//		start = MPI_Wtime();
		std::pair<coor_t , coor_t > result = compute_difference_and_error();
//		end = MPI_Wtime();
//		std::cout << "computed difference and error (" << end - start << ")" << std::endl;
		return result;
	}


	void launch() {
		const int MAX_ITERATION = 1000;
		int iteration = 0;
		for (; iteration < MAX_ITERATION ;++iteration) {
			double start = MPI_Wtime();
			std::pair<coor_t, coor_t> result = calculate_iteration();
			double end = MPI_Wtime();
			coor_t difference = result.first;
			coor_t error = result.second;
			std::cout << "[" << iteration << "] " << difference << " " << error
			          << "(" << end - start << ")" << std::endl;
			if (difference < eps)
				break;
//			std::cout << "p == new_p is " << (cur == next? "true" : "false") << std::endl;
			std::cout << "cur  p " << cur.p << std::endl;
			std::cout << "next p " << next.p << std::endl;
//			std::cout << "r " << next.r << std::endl;
//			std::cout << "g " << next.g << std::endl;
			cur = next;
		}
	}

	~LocalProcess() {
		delete up_neighbor;
		delete down_neighbor;
		delete left_neighbor;
		delete right_neighbor;
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



	const int N = 3;


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
	OneDimensionData x_data = OneDimensionData(N, 0, 2, 3.0 / 2, x_range.first, x_range.second);
	OneDimensionData y_data = OneDimensionData(N, 0, 2, 3.0 / 2, y_range.first, y_range.second);

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