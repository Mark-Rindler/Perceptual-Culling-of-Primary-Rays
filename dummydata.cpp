#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <cstdint>
#include <algorithm>

struct Vec3 {
    float x, y, z;
};

inline Vec3 make_vec3(float x, float y, float z) {
    return {x, y, z};
}

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(const Vec3& a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}

inline Vec3 operator/(const Vec3& a, float s) {
    return {a.x / s, a.y / s, a.z / s};
}

inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

inline float length(const Vec3& a) {
    return std::sqrt(dot(a, a));
}

inline Vec3 normalize(const Vec3& a) {
    float len = length(a);
    if (len > 1e-8f) return a / len;
    return {0.0f, 0.0f, 1.0f};
}


bool rayIntersectsTriangle(
    const Vec3& orig,
    const Vec3& dir,
    const Vec3& v0,
    const Vec3& v1,
    const Vec3& v2
) {
    const float EPSILON = 1e-7f;
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    Vec3 h = cross(dir, edge2);
    float a = dot(edge1, h);
    if (std::fabs(a) < EPSILON)
        return false; // Parallel

    float f = 1.0f / a;
    Vec3 s = orig - v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f)
        return false;
    Vec3 q = cross(s, edge1);
    float v = f * dot(dir, q);
    if (v < 0.0f || u + v > 1.0f)
        return false;
    float t = f * dot(edge2, q);
    if (t > EPSILON) // intersection
        return true;
    return false;
}

// TriangleSample struct 

struct TriangleSample {
    //RAY
    float ray_dir[3];      // 3

    //TRIANGLE GEOMETRY
    float v0[3];           // 3
    float v1[3];           // 3
    float v2[3];           // 3

    float perimeter;        // 1
    float area;             // 1
    float unit_normal[3];   // 3
    float plane_eq[4];      // 4: ax + by + cz + d
    float orientation_sign; // 1
    float aspect_ratio;     // 1
    float barycentric_condition; // 1 (dummy but consistent)

    float centroid[3];      // 3
    float orthocenter[3];   // 3 (reuse centroid)
    float fermat_point[3];  // 3 (reuse centroid)

    float circumcenter[3];  // 3
    float circumradius;     // 1
    float incenter[3];      // 3
    float inradius;         // 1

    // --- BOUNDING BOXES ---
    float aabb[6];          // 6: min(x,y,z), max(x,y,z)
    float obb[10];          // 10: center(3), extents(3), normal(3), padding(1)

    // --- LABEL ---
    uint8_t label;          // 1 = hit, 0 = miss
};

// Helper to fill TriangleSample from triangle + ray
TriangleSample makeSample(const Vec3& rayOrig,
                          const Vec3& rayDir,
                          const Vec3& v0,
                          const Vec3& v1,
                          const Vec3& v2)
{
    TriangleSample s{}; // value-initialize everything to 0

    // Ray (direction only — origin is NOT stored)
    s.ray_dir[0] = rayDir.x;
    s.ray_dir[1] = rayDir.y;
    s.ray_dir[2] = rayDir.z;

    // Vertices
    s.v0[0] = v0.x; s.v0[1] = v0.y; s.v0[2] = v0.z;
    s.v1[0] = v1.x; s.v1[1] = v1.y; s.v1[2] = v1.z;
    s.v2[0] = v2.x; s.v2[1] = v2.y; s.v2[2] = v2.z;

    // Edges
    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;

    float len0 = length(e0);
    float len1 = length(e1);
    float len2 = length(e2);

    s.perimeter = len0 + len1 + len2;

    // Area
    Vec3 n = cross(v1 - v0, v2 - v0);
    float area = 0.5f * length(n);
    s.area = area;

    // Normal
    Vec3 unitN = normalize(n);
    s.unit_normal[0] = unitN.x;
    s.unit_normal[1] = unitN.y;
    s.unit_normal[2] = unitN.z;

    // Plane eq: n.x * x + n.y * y + n.z * z + d = 0
    float d = -dot(unitN, v0);
    s.plane_eq[0] = unitN.x;
    s.plane_eq[1] = unitN.y;
    s.plane_eq[2] = unitN.z;
    s.plane_eq[3] = d;

    // Orientation sign (front-facing vs back-facing relative to ray)
    float orient = dot(unitN, rayDir);
    s.orientation_sign = (orient >= 0.0f) ? 1.0f : -1.0f;

    // Aspect ratio: max edge length / min edge length
    float maxEdge = std::max(len0, std::max(len1, len2));
    float minEdge = std::min(len0, std::min(len1, len2));
    if (minEdge < 1e-6f) minEdge = 1e-6f;
    s.aspect_ratio = maxEdge / minEdge;

    // Dummy "condition number" based on edge lengths (not rigorous)
    s.barycentric_condition = s.aspect_ratio * s.aspect_ratio;

    // Centroid
    Vec3 centroid = (v0 + v1 + v2) / 3.0f;
    s.centroid[0] = centroid.x;
    s.centroid[1] = centroid.y;
    s.centroid[2] = centroid.z;

    // For dummy data, reuse centroid as orthocenter & Fermat point
    s.orthocenter[0] = centroid.x;
    s.orthocenter[1] = centroid.y;
    s.orthocenter[2] = centroid.z;

    s.fermat_point[0] = centroid.x;
    s.fermat_point[1] = centroid.y;
    s.fermat_point[2] = centroid.z;

    float r0 = length(v0 - centroid);
    float r1 = length(v1 - centroid);
    float r2 = length(v2 - centroid);
    float avgR = (r0 + r1 + r2) / 3.0f;

    s.circumcenter[0] = centroid.x;
    s.circumcenter[1] = centroid.y;
    s.circumcenter[2] = centroid.z;
    s.circumradius = avgR;

    // Incenter: dummy = centroid; inradius = area / (0.5 * perimeter)
    s.incenter[0] = centroid.x;
    s.incenter[1] = centroid.y;
    s.incenter[2] = centroid.z;
    if (s.perimeter > 1e-6f)
        s.inradius = (2.0f * area) / s.perimeter;
    else
        s.inradius = 0.0f;

    // Axis-aligned bounding box
    float minx = std::min(v0.x, std::min(v1.x, v2.x));
    float miny = std::min(v0.y, std::min(v1.y, v2.y));
    float minz = std::min(v0.z, std::min(v1.z, v2.z));
    float maxx = std::max(v0.x, std::max(v1.x, v2.x));
    float maxy = std::max(v0.y, std::max(v1.y, v2.y));
    float maxz = std::max(v0.z, std::max(v1.z, v2.z));

    s.aabb[0] = minx;
    s.aabb[1] = miny;
    s.aabb[2] = minz;
    s.aabb[3] = maxx;
    s.aabb[4] = maxy;
    s.aabb[5] = maxz;

    // Dummy oriented bounding box (OBB):
    // center(3), extents(3), normal(3), padding(1) = 10 floats
    Vec3 center = centroid;
    Vec3 extents = make_vec3(
        0.5f * (maxx - minx),
        0.5f * (maxy - miny),
        0.5f * (maxz - minz)
    );
    Vec3 obbNormal = unitN; // reuse triangle normal

    s.obb[0] = center.x;
    s.obb[1] = center.y;
    s.obb[2] = center.z;
    s.obb[3] = extents.x;
    s.obb[4] = extents.y;
    s.obb[5] = extents.z;
    s.obb[6] = obbNormal.x;
    s.obb[7] = obbNormal.y;
    s.obb[8] = obbNormal.z;
    s.obb[9] = 0.0f; // padding / reserved

    // Label: hit or miss
    bool hit = rayIntersectsTriangle(rayOrig, rayDir, v0, v1, v2);
    s.label = hit ? 1 : 0;

    return s;
}

// ----------------------
// Helpers for sampling near the ray
// ----------------------

// Build an orthonormal basis (u, v) around direction w = dir
inline void buildOrthonormalBasis(const Vec3& dir, Vec3& u, Vec3& v) {
    Vec3 w = normalize(dir);
    Vec3 tmp = (std::fabs(w.x) < 0.9f) ? make_vec3(1.0f, 0.0f, 0.0f)
                                       : make_vec3(0.0f, 1.0f, 0.0f);
    u = normalize(cross(w, tmp)); // perpendicular to w
    v = cross(w, u);              // completes RHS basis
}

// Generate a small triangle near the ray; if makeHit==true, it will be
// centered exactly on the ray so the ray goes through its interior.

template<typename RNG>
void generateTriangleNearRay(
    const Vec3& rayOrigin,
    const Vec3& rayDir,
    RNG& gen,
    std::uniform_real_distribution<float>& tDist,
    std::uniform_real_distribution<float>& sizeDist,
    std::uniform_real_distribution<float>& offsetDist,
    std::uniform_real_distribution<float>& angleDist,
    bool makeHit,
    Vec3& out_v0,
    Vec3& out_v1,
    Vec3& out_v2
) {
    // Ray point in front of origin
    float t = tDist(gen);
    Vec3 pointOnRay = rayOrigin + rayDir * t;

    // Local basis around ray
    Vec3 u, v;
    buildOrthonormalBasis(rayDir, u, v);

    // Triangle size (kept small so vertices stay close to the ray)
    float size = sizeDist(gen); // ~0.05 to 0.15

    // Center of triangle
    Vec3 center = pointOnRay;

    if (!makeHit) {
        // Move center off the ray in a random perpendicular direction,
        // but keep the offset small so distance to ray <= ~0.5.
        float off   = offsetDist(gen);      // up to ~0.2 (see main)
        float theta = angleDist(gen);       // [0, 2π)
        float c = std::cos(theta);
        float s = std::sin(theta);
        Vec3 offset = u * (off * c) + v * (off * s);
        center = pointOnRay + offset;
    }

    // Make a non-degenerate triangle around 'center' in the (u, v) plane
    // so that center is exactly its centroid.
    //
    // v0 = center + u*size + v*size
    // v1 = center - u*size + v*size
    // v2 = center - 2*v*size
    //
    // (v0 + v1 + v2) / 3 = center
    out_v0 = center + u * size + v * size;
    out_v1 = center - u * size + v * size;
    out_v2 = center - v * (2.0f * size);
}

// ----------------------
// Main: generate data
// ----------------------

int main() {
    const int   NUM_SAMPLES     = 10000;
    const float TARGET_HIT_PROB = 0.13f; // probability we *try* to make a hit

    // Binary output file
    std::ofstream out("triangles.bin", std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output file.\n";
        return 1;
    }

    // Random generator
    std::random_device rd;
    std::mt19937      gen(rd());

    // Random ray directions on unit sphere
    std::uniform_real_distribution<float> dirDist(-1.0f, 1.0f);

    // Ray parameter along the ray (where triangles are placed)
    std::uniform_real_distribution<float> tDist(0.5f, 10.0f);

    // Triangle size: small so vertices are close to the ray
    std::uniform_real_distribution<float> sizeDist(0.05f, 0.15f);

    // Perpendicular offset for "miss" triangles (controls max distance)
    // With offset <= 0.2 and size <= 0.15, max distance to ray is <= ~0.5
    std::uniform_real_distribution<float> offsetDist(0.05f, 0.2f);

    // Angle for offset direction around the ray
    const float PI = 3.14159265358979323846f;
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * PI);

    // For choosing hits vs misses
    std::uniform_real_distribution<float> probDist(0.0f, 1.0f);

    // Ray origin fixed at (0,0,0) but **not** stored in TriangleSample
    Vec3 rayOrigin = make_vec3(0.0f, 0.0f, 0.0f);

    int labelHitCount = 0;

    for (int i = 0; i < NUM_SAMPLES; ++i) {
        // Random ray direction
        Vec3 dir;
        do {
            dir = make_vec3(
                dirDist(gen),
                dirDist(gen),
                dirDist(gen)
            );
        } while (length(dir) < 1e-3f);
        dir = normalize(dir);

        // Decide if we want this triangle to be a "hit" or a "near miss"
        bool makeHit = (probDist(gen) < TARGET_HIT_PROB);

        Vec3 v0, v1, v2;
        generateTriangleNearRay(
            rayOrigin,
            dir,
            gen,
            tDist,
            sizeDist,
            offsetDist,
            angleDist,
            makeHit,
            v0, v1, v2
        );

        TriangleSample sample = makeSample(rayOrigin, dir, v0, v1, v2);
        if (sample.label == 1) {
            ++labelHitCount;
        }

        out.write(reinterpret_cast<const char*>(&sample), sizeof(TriangleSample));
    }

    out.close();

    float actualHitFrac = static_cast<float>(labelHitCount) / NUM_SAMPLES;
    std::cout << "Wrote " << NUM_SAMPLES << " TriangleSample records to triangles.bin\n";
    std::cout << "Actual hit fraction (from Moller-Trumbore): " << actualHitFrac << "\n";

    return 0;
}
