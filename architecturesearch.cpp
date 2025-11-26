#include <cstdint> 
#include <fstream>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

// I manually removed all cases where there was 1 node per layer and multiple layers!!!

struct Row {
    int     n;      // input size
    int     L;      // hidden-layer count
    int     H;      // hidden size
    int64_t flops;  // forward-pass FLOPS 
};

static inline int64_t flops(int n, int L, int H)
{
    int64_t w_in  = int64_t(n)       * H;      
    int64_t w_hid = int64_t(L - 1)   * H * H;  
    int64_t w_out = int64_t(H);                
    return 2 * (w_in + w_hid + w_out);
}

int main()
{
    //search limits
    const int MAX_N      = 150;
    const int MAX_LAYERS = 9999;
    const int MAX_H      = 9999;

    // SCALED FLOPS LIMITS
    const int64_t MAX_FLOPS_T1 = 60;  // Base (was 600)
    const int64_t MAX_FLOPS_T2 = 72;  // (was 720)
    const int64_t MAX_FLOPS_T3 = 81;  // (was 810)
    const int64_t MAX_FLOPS_T4 = 100; // (was 1000)

    // Optimization limit is the highest possible value
    const int64_t MAX_OPTIMIZATION_LIMIT = MAX_FLOPS_T4; 

    std::vector<Row> T1, T2, T3, T4;

    for (int n = 7; n <= MAX_N; ++n) {
        int64_t min_f_for_n = flops(n, 1, 1);
        if (min_f_for_n < 0 || min_f_for_n >= MAX_OPTIMIZATION_LIMIT) {
            break; 
        }

        for (int L = 1; L <= MAX_LAYERS; ++L) {
            int64_t min_f_for_L = flops(n, L, 1); 
            if (min_f_for_L < 0 || min_f_for_L >= MAX_OPTIMIZATION_LIMIT) {
                break; // All subsequent L will also be too large
            }

            for (int H = 1; H <= MAX_H; ++H) { // Starts from H=1
                int64_t f = flops(n, L, H);

                if (f < 0 || f >= MAX_OPTIMIZATION_LIMIT) {
                    break; // All subsequent H will also be too large
                }

                if (f < MAX_FLOPS_T1) {                               
                    T1.push_back({n, L, H, f});
                } else if (n % 4 == 0 && f < MAX_FLOPS_T2) {          
                    T2.push_back({n, L, H, f});
                } else if (L % 8 == 0 && 
                           f < MAX_FLOPS_T3) {
                    T3.push_back({n, L, H, f});
                } else if (n % 4 == 0 &&
                           H % 8 == 0 && 
                           f < MAX_FLOPS_T4) {                       
                    T4.push_back({n, L, H, f});
                }
            }
        }
    }
    
    std::ofstream out("nn_flops_tables.txt");
    if (!out) {
        std::cerr << "Could not open output file.\n";
        return 1;
    }

    auto dump = [&](const std::vector<Row>& v, const std::string& title) {
        out << title << " (n  L  H  FLOPS)\n"
            << "-----------------------------------\n";
        for (const auto& r : v)
            out << std::setw(3)  << r.n     << ' '
                << std::setw(3)  << r.L     << ' '
                << std::setw(4)  << r.H     << ' '
                << std::setw(6)  << r.flops << '\n';
        out << '\n';
    };


    const std::string t1_title = "Table 1: n>6, FLOPS<60";
    dump(T1, t1_title);

    const std::string t2_title = "Table 2: 4|n, FLOPS<72, not in T1";
    dump(T2, t2_title);

    const std::string t3_title = "Table 3: 8|L, FLOPS<81, not in T1-T2";
    dump(T3, t3_title);

    const std::string t4_title = "Table 4: 4|n & 8|H, FLOPS<100, unique";
    dump(T4, t4_title);

    std::cout << "Done.  "
              << (T1.size() + T2.size() + T3.size() + T4.size())
              << " rows written to nn_flops_tables.txt\n";
    return 0;
}