#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string> // Added for std::string
#include <tuple>
#include <vector>

struct Row {
    int     n;      // input size
    int     L;      // hidden-layer count
    int     H;      // hidden size
    int64_t flops;  // forward-pass FLOPS
};

/**
 * @brief Calculates the forward-pass FLOPS for a simple MLP.
 */
static inline int64_t flops(int n, int L, int H)
{
    // 2 * ( (n*H) + (L-1)*(H*H) + (H*1) )
    int64_t w_in  = int64_t(n) * H;
    int64_t w_hid = int64_t(L - 1) * H * H;
    int64_t w_out = int64_t(H);
    return 2 * (w_in + w_hid + w_out);
}

// Helper limits struct
struct FlopsLimits {
    const int64_t T1;
    const int64_t T2;
    const int64_t T3;
    const int64_t T4;
    const int64_t MAX_OPTIMIZATION_LIMIT;
};

/**
 * @brief Checks a model's FLOPS and adds it to the appropriate table.
 * * This implements the cascading if/else if logic, ensuring a model
 * only belongs to the first table it qualifies for.
 */
void categorize_model(int n, int L, int H, int64_t f,
                      const FlopsLimits& limits,
                      std::vector<Row>& T1, std::vector<Row>& T2,
                      std::vector<Row>& T3, std::vector<Row>& T4)
{
    if (f < limits.T1) {
        T1.push_back({n, L, H, f});
    } else if (n % 4 == 0 && f < limits.T2) {
        T2.push_back({n, L, H, f});
    } else if (L % 8 == 0 && f < limits.T3) {
        T3.push_back({n, L, H, f});
    } else if (n % 4 == 0 && H % 8 == 0 && f < limits.T4) {
        T4.push_back({n, L, H, f});
    }
}

/**
 * @brief Dumps a vector of Row structs to the output file stream.
 */
void dump_table(std::ofstream& out, const std::vector<Row>& v, const std::string& title)
{
    out << title << " (n  L  H  FLOPS)\n"
        << "-----------------------------------\n";
    for (const auto& r : v)
        out << std::setw(3) << r.n << ' '
            << std::setw(3) << r.L << ' '
            << std::setw(4) << r.H << ' '
            << std::setw(6) << r.flops << '\n';
    out << '\n';
}


int main()
{
    // Search limits
    const int MAX_N      = 256;
    const int MAX_LAYERS = 9999;
    const int MAX_H      = 9999;

    // SCALED FLOPS LIMITS
    const FlopsLimits limits = {
        .T1 = 45,  // Base (was 450)
        .T2 = 54,  // (was 540)
        .T3 = 60,  // (was 600)
        .T4 = 75,  // (was 750)
        .MAX_OPTIMIZATION_LIMIT = 75 // Must be the same as the highest limit (T4)
    };

    std::vector<Row> T1, T2, T3, T4;

    for (int n = 4; n <= MAX_N; ++n) {
        // Optimization: check the smallest possible model for this 'n'.
        // If (n, 1, 1) is already too large, all other models for this
        // 'n' will also be too large, so we can stop.
        int64_t min_f_for_n = flops(n, 1, 1);
        if (min_f_for_n < 0 || min_f_for_n >= limits.MAX_OPTIMIZATION_LIMIT) {
            break; // Stop iterating on 'n'
        }

        // --- Case 1: L = 1 ---
        // This is the *only* case where H=1 is allowed.
        // H can be 1, 2, 3, ...
        {
            const int L = 1;
            for (int H = 1; H <= MAX_H; ++H) {
                int64_t f = flops(n, L, H);

                if (f < 0 || f >= limits.MAX_OPTIMIZATION_LIMIT) {
                    break; // All subsequent H will also be too large
                }
                
                categorize_model(n, L, H, f, limits, T1, T2, T3, T4);
            }
        } // End of L=1 case

        // --- Case 2: L > 1 ---
        // As per the constraint, if L > 1, H *must* also be > 1.
        // Therefore, L starts at 2, and H also starts at 2.
        for (int L = 2; L <= MAX_LAYERS; ++L) {
            // Optimization: check smallest model for this L (which is H=2)
            int64_t min_f_for_L = flops(n, L, 2);
            if (min_f_for_L < 0 || min_f_for_L >= limits.MAX_OPTIMIZATION_LIMIT) {
                break; // All subsequent L will also be too large
            }

            // Start H from 2, enforcing the (L > 1) => (H > 1) rule
            for (int H = 2; H <= MAX_H; ++H) {
                int64_t f = flops(n, L, H);

                if (f < 0 || f >= limits.MAX_OPTIMIZATION_LIMIT) {
                    break; // All subsequent H will also be too large
                }

                categorize_model(n, L, H, f, limits, T1, T2, T3, T4);
            }
        } // End of L > 1 case
    }
    
    std::ofstream out("nn_flops_tables.txt");
    if (!out) {
        std::cerr << "Could not open output file.\n";
        return 1;
    }

    dump_table(out, T1, "Table 1: 4<=n<=6, FLOPS<45");
    dump_table(out, T2, "Table 2: 4|n, FLOPS<54, not in T1");
    dump_table(out, T3, "Table 3: 8|L, FLOPS<60, not in T1-T2");
    dump_table(out, T4, "Table 4: 4|n & 8|H, FLOPS<75, unique");

    std::cout << "Done.  "
              << (T1.size() + T2.size() + T3.size() + T4.size())
              << " rows written to nn_flops_tables.txt\n";
    return 0;
}