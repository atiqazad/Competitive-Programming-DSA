#include<bits/stdc++.h>
using namespace std;

// -------------------- Typedefs --------------------
typedef pair<int, int> pii;
typedef vector<string> vs;
typedef vector<long long> vi;
typedef vector<char> vc;
typedef long long ll;
#define INF 1e18
#define all(x) x.begin(), x.end()
#define sq(a) (a)*(a)
#define pb push_back
#define pob pop_back
#define MOD 1000000007
#define sz(x) (int)x.size()
#define int long long
#define setBits(x) __builtin_popcountll(x)
#define get_binary_length(n) (64-__builtin_clzll(n))
#define setBitToOne(ans, pos) ans |= (1LL << pos)

const int N = 1e5 + 5;

#define LOCAL  // for local only
#ifdef LOCAL
#define debug(x) cerr << #x << " = "; _print(x); cerr << endl;
#else
#define debug(x)
#endif
// Overloads of _print for common types

void _print(int x) { cerr << x; }
// void _print(long long x) { cerr << x; }
void _print(char x) { cerr << '\'' << x << '\''; }
void _print(const string &x) { cerr << '\"' << x << '\"'; }
void _print(bool x) { cerr << (x ? "true" : "false"); }

template<typename T, typename V>
void _print(const pair<T, V> &p) {
    cerr << '{'; _print(p.first); cerr << ','; _print(p.second); cerr << '}';
}

template<typename T>
void _print(const vector<T> &v) {
    cerr << "[ ";
    for (const auto &item : v) {
        _print(item);
        cerr << " ";
    }
    cerr << "]";
}

template<typename T>
void _print(const set<T> &s) {
    cerr << "{ ";
    for (const auto &item : s) {
        _print(item);
        cerr << " ";
    }
    cerr << "}";
}

template<typename T, typename V>
void _print(const map<T, V> &m) {
    cerr << "{ ";
    for (const auto &item : m) {
        _print(item);
        cerr << " ";
    }
    cerr << "}";
}
//remember pow() always returns double type, so like compares like
/*
Use Dijkstra’s algorithm for shortest paths with positive weights.

Use Floyd-Warshall for all-pairs shortest paths (good for smaller graphs).

Use Bellman-Ford if negative weights or negative cycles are involved.
*/

// -------------------- Lambda Functions --------------------
// Anonymous functions (can capture variables and be passed inline)

// Example: Simple lambda to add 2 numbers
// auto add = [](int a, int b) {
//     return a + b;
// };
// cout << add(2, 3); // → 5

// -------------------- Comparator Functions --------------------
// For sorting in custom order

// Sort pairs by second value (ascending)
// bool cmp(pii a, pii b) {
//     return a.second < b.second;
// }

// Or use lambda directly:
// auto cmp_lambda = [](pii a, pii b) {
//     return a.second < b.second;
// };

// -------------------- Sorting Vectors --------------------
// vector<pii> vp = {{1, 3}, {2, 2}, {4, 1}};

// sort(all(vp), cmp);         // using function
// or
// sort(all(vp), cmp_lambda);  // using lambda

// Sort only a part of a vector
// vector<int> v = {5, 1, 3, 2, 4};
// sort(v.begin() + 1, v.begin() + 4);  // sort index 1 to 3


// -------------------- 2D Priority Queue (pair or tuple) --------------------

// Max-Heap by default (based on first element)
// priority_queue<pii> pq1;

// Min-Heap: sort by first element ascending
// priority_queue<pii, vector<pii>, greater<pii>> pq2;

// If you want to sort by second element ascending:
// auto cmp_pair = [](pii a, pii b) {
//     return a.second > b.second;  // min-heap by second
// };
// priority_queue<pii, vector<pii>, decltype(cmp_pair)> pq3(cmp_pair);

// Example: pushing into 2D priority queue
// pq3.push({10, 5});
// pq3.push({3, 8});

// -------------------- Tuple Priority Queue --------------------
// typedef tuple<int, int, int> t3;

// Sort by first, then second, then third (min-heap)
// auto cmp_tuple = [](t3 a, t3 b) {
//     if (get<0>(a) != get<0>(b)) return get<0>(a) > get<0>(b);  // sort by first
//     if (get<1>(a) != get<1>(b)) return get<1>(a) > get<1>(b);  // then second
//     return get<2>(a) > get<2>(b);                              // then third
// };

// priority_queue<t3, vector<t3>, decltype(cmp_tuple)> pq4(cmp_tuple);

// Example usage:
// pq4.push({2, 5, 1});
// pq4.push({2, 4, 7});

// -------------------- Disjoint Set Union (Union Find) --------------------
class UnionFind {
    vi parent, rank;
public:
    UnionFind(int n): parent(n+1), rank(n+1, 1) {
        iota(parent.begin(), parent.end(), 0);
    }
    int find(int x) {
        return (x == parent[x]) ? x : parent[x] = find(parent[x]);
    }

    void unite(int x, int y) {
        x = find(x), y = find(y);
        if(x == y) return;
        else if(rank[x] > rank[y]) {
            parent[y] = x;
            rank[x] += rank[y];
        }
        else {
            parent[x] = y;
            rank[y] += rank[x];
        }

    }
};
struct DSU {
    vi parent, size;
    DSU(int n) {
        parent.resize(n + 1);
        size.assign(n + 1, 1);
        iota(parent.begin(), parent.end(), 0); // parent[i] = i
    }

    int find(int x) {
        if (x == parent[x]) return x;
        return parent[x] = find(parent[x]); // Path compression
    }

    bool unite(int x, int y) {
        x = find(x), y = find(y);
        if (x == y) return false;
        if (size[x] < size[y]) swap(x, y);
        parent[y] = x;
        size[x] += size[y];
        return true;
    }
};

// -------------------- Graph (DFS and BFS) --------------------

// vector<int> adj[N];
// bool visited[N];

// void dfs(int u) {
//     visited[u] = true;
//     for (int v : adj[u]) {
//         if (!visited[v]) dfs(v);
//     }
// }

// void bfs(int start) {
//     queue<int> q;
//     q.push(start);
//     visited[start] = true;
//     while (!q.empty()) {
//         int u = q.front(); q.pop();
//         for (int v : adj[u]) {
//             if (!visited[v]) {
//                 visited[v] = true;
//                 q.push(v);
//             }
//         }
//     }
// }

// vector<pii> adj[N];  // adjacency list: adj[u] = {v, weight}
// vector<int> dist(N, INF);  // shortest distance from source
// vector<bool> visited(N, false);

// void dijkstra(int source) {
//     // Min-heap priority queue: {distance, node}
//     priority_queue<pii, vector<pii>, greater<pii>> pq;
//     dist[source] = 0;
//     pq.push({0, source});  // Start with source node

//     while (!pq.empty()) {
//         int d = pq.top().first;
//         int u = pq.top().second;
//         pq.pop();

//         if (visited[u]) continue;
//         visited[u] = true;

//         for (auto [v, w] : adj[u]) {
//             if (dist[v] > d + w) {
//                 dist[v] = d + w;
//                 pq.push({dist[v], v});  // Push updated dist to PQ
//             }
//         }
//     }
// }

// floyd-warshall Given a directed or an undirected weighted graph  G with  n  vertices. 
// The task is to find the length of the shortest path  d[i][j]  between each pair of vertices i  and  j .
// The graph may have negative weight edges, but no negative weight cycles.
// vector<vector<int>> d
// Let  d[][]  is a 2D array of size  nXn , 
// which is filled according to the  0 -th phase as explained earlier. 
// Also we will set  d[i][i] = 0  for any  i  at the  0 -th phase.

void floydWarshall(vector<vector<int>>& d, int n) {
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]); 
            }
        }
    }

    // However if there are negative weight edges in the graph, special measures have to be taken. 
    // Otherwise the resulting values in matrix may be of the form inf - 1, inf - 2, etc., 
    // which, of course, still indicates that between the respective vertices doesn't exist a path. 
    // Therefore, if the graph has negative weight edges, 
    // it is better to write the Floyd-Warshall algorithm in the following way, so
    // that it does not perform transitions using paths that don't exist.
    // for (int k = 0; k < n; ++k) {
    //     for (int i = 0; i < n; ++i) {
    //         for (int j = 0; j < n; ++j) {
    //             if (d[i][k] < INF && d[k][j] < INF)
    //                 d[i][j] = min(d[i][j], d[i][k] + d[k][j]); 
    //         }
    //     }
    // }

}



// -------------------- Modular Arithmetic --------------------

int mod_add(int a, int b) {
    return (a % MOD + b % MOD) % MOD;
}

int mod_sub(int a, int b) {
    return (a % MOD - b % MOD + MOD) % MOD;
}

int mod_mul(int a, int b) {
    return (a % MOD * b % MOD) % MOD;
}

int mod_pow(int a, int b) {
    int res = 1;
    a %= MOD;
    while (b > 0) {
        if (b & 1) res = mod_mul(res, a);
        a = mod_mul(a, a);
        b >>= 1;
    }
    return res;
}

int mod_inv(int a) {
    return mod_pow(a, MOD - 2);  // Fermat's little theorem (if MOD is prime)
}

int mod_div(int a, int b) {
    return mod_mul(a, mod_inv(b));
}

// -------------------- Binary Search Template --------------------
int binary_search_example(vi &arr, int target) {
    int lo = 0, hi = arr.size() - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1; // Not found
}

// -------------------- lower_bound & upper_bound --------------------
// Works on: vector, set, multiset (sorted containers)

// In VECTORS:
// lower_bound(vec.begin(), vec.end(), x) → first element ≥ x
// upper_bound(vec.begin(), vec.end(), x) → first element > x

// Example:
// vector<int> v = {1, 2, 4, 4, 5, 6};
// auto it = lower_bound(all(v), 4);  // points to first 4
// auto it = upper_bound(all(v), 4);  // points to 5

// In SETS:
// s.lower_bound(x) → iterator to first element ≥ x
// s.upper_bound(x) → iterator to first element > x

// Example:
// set<int> s = {1, 3, 5, 7};
// auto it = s.lower_bound(4);  // points to 5
// auto it = s.upper_bound(5);  // points to 7

// To get index from iterator in vector:
// int idx = it - v.begin();  // index of found element


bool isAllBitsOn(int n) {
    return n == (1LL << get_binary_length(n)) - 1;
}
int rightMostZeroBit(int n) {
    int pos = 0;
    while ((n & (1LL << pos)) != 0) {
        pos++;
    }
    return pos;
}
//both decToBin & binToDec are slow functions 
//always use bit operations if the problem requires
//to transform the decimal to binary or vice-versa.
string decToBin(int num) {
    
    if(num == 0) return "0";
    string ans = "";
    while(num > 0) {
        int remain = num%2;
        char r = '0' + remain;
        ans = r + ans;
        num /= 2;
    }

    return ans;
}

int binToDec(string s) {
    int pow = 1;
    int ans = 0;
    for(int j = s.size()-1;j >= 0;j--) {
        ans += (pow*(s[j]-'0'));
        pow *= 2;
    }
    return ans;
}
int modularBinaryExponential(int base, int exponent) {
    if(exponent == 0) return 1;
    int result = modularBinaryExponential(base, exponent/2);
    if(exponent%2 == 1) {
        return (((result*result)%MOD)*base)%MOD;
    }
    else return (result*result)%MOD;
}

int maxSubArraySum(vector<int>& nums) {
    int max_sum = INT_MIN;
    int current_sum = 0;

    for (int x : nums) {
        current_sum = max(x, current_sum + x);
        max_sum = max(max_sum, current_sum);
    }

    return max_sum;
}


// -------------------- Fast IO --------------------
#define fast_io() ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);



// -------------------- Conversion Utilities --------------------
string to_str(int num) { return to_string(num); }
int to_int(const string &s) { return stoll(s); }
char digit_to_char(int d) { return d + '0'; }
int char_to_digit(char c) { return c - '0'; }

// -------------------- Priority Queues --------------------
priority_queue<int> maxHeap;
priority_queue<int, vector<int>, greater<int>> minHeap;

// -------------------- GCD/LCM -----------------
int gcd(int a, int b) {return b == 0 ? a : gcd(b, a % b);}

int lcm(int a, int b) { return a / gcd(a, b) * b; }

// -------------- FractionToDecimal ----------------

string fractionToDecimal(int numerator, int denominator) {
    if(numerator == 0) return "0";
    if(denominator == 0) return "Divison By Zero";
    string fraction = "";
    long long numeratorr = numerator;
    long long denominatorr = denominator;
    long long dividend = abs(numeratorr)/abs(denominatorr);
    long long divisor = abs(denominatorr);
    long long remainder = abs(numeratorr)%abs(denominatorr);
    if((numeratorr < 0) ^ (denominatorr < 0)){
        fraction += '-';
    }
    // if(numerator < 0 && denominator/)
    fraction += to_string(dividend);
    if(remainder == 0) {
        return fraction;
    }   
    fraction += '.';
    unordered_map<int, int> mp; 
    // remainder *
    while(remainder != 0) {
        if(mp.count(remainder)) {
            fraction.insert(mp[remainder], "(");
            fraction += ')';
            break;
        }
        mp[remainder] = fraction.size();
        remainder *= 10;
        fraction += to_string(remainder/divisor);
        remainder = remainder%divisor;
    }
    return fraction;
}



// -------------------- Solve -------------------
void solve(int ind) {
    int n;
    cin >> n;
    vi a(n), b(n);
    for(int i = 0;i < n;i++) cin >> a[i];
    for(int i = 0;i < n;i++) cin >> b[i];
    sort(a.rbegin(), a.rend());
    map<int,int> mp;
    for(int i = 0;i < n;i++) {
        if(i > 0 && a[i] != a[i-1]) {
            mp[a[i]] = 1 + mp[a[i-1]];
        }
        else {
            mp[a[i]]++;
        }
    }
    vi pre_sum(n);
    pre_sum[0] = b[0];
    for(int i = 1;i < n;i++) {
        pre_sum[i] = b[i] + pre_sum[i-1];
    }
    int max_score = 0;
    for(auto& p: mp) {
        // cout << p.first << " " << p.second << endl;
        int total_strike = p.second;
        int l = 0;
        int r = n-1;
        int level = -1;
        // int ans = -1;
        while(l <= r) {
            int mid = l + (r-l)/2;
            if(pre_sum[mid] == total_strike) {
                level = mid;
                break;
            }
            else if(pre_sum[mid] > total_strike) {
                r = mid-1;
            }
            else {
                level = mid;
                l = mid+1;
            }
           
        }
        
        // cout << p.first << " " << p.second << " " << level << endl;
        
        max_score = max(max_score, p.first*(level+1));
    }
    cout << max_score;
}



// -------------------- Main --------------------

int32_t main()
{
    fast_io();
    int tt=1, k = 1;
    cin >> tt;
    while(tt--) {
        solve(k);
        k++;
        cout << "\n";
    }

    /*while(scanf("%d %d", &a, &b)!=EOF){
            //fill it in
    }*/
    //solve(0);
    //exception_handle();

    /*The Boyer-Moore Majority Voting Algorithm, efficiently finds a majority element (if it exists) in linear time without using extra space.*/

}




