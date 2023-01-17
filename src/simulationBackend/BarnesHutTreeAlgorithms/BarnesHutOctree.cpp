#include "BarnesHutOctree.hpp"
#include "Configuration.hpp"

BarnesHutOctree::BarnesHutOctree() :
        octants_vec(8 * configuration::barnes_hut_algorithm::storageSizeParameter),
        octants(octants_vec.data(), octants_vec.size()),

        edgeLengths_vec(configuration::barnes_hut_algorithm::storageSizeParameter),
        edgeLengths(edgeLengths_vec.data(), edgeLengths_vec.size()),

        min_x_values_vec(configuration::barnes_hut_algorithm::storageSizeParameter),
        min_y_values_vec(configuration::barnes_hut_algorithm::storageSizeParameter),
        min_z_values_vec(configuration::barnes_hut_algorithm::storageSizeParameter),
        min_x_values(min_x_values_vec.data(), min_x_values_vec.size()),
        min_y_values(min_y_values_vec.data(), min_y_values_vec.size()),
        min_z_values(min_z_values_vec.data(), min_z_values_vec.size()),

        bodyOfNode_vec(configuration::barnes_hut_algorithm::storageSizeParameter, numberOfBodies),
        bodyOfNode(bodyOfNode_vec.data(), bodyOfNode_vec.size()),

        sumMasses_vec(configuration::barnes_hut_algorithm::storageSizeParameter),
        centerOfMass_x_vec(configuration::barnes_hut_algorithm::storageSizeParameter),
        centerOfMass_y_vec(configuration::barnes_hut_algorithm::storageSizeParameter),
        centerOfMass_z_vec(configuration::barnes_hut_algorithm::storageSizeParameter),
        sumOfMasses(sumMasses_vec.data(), sumMasses_vec.size()),
        massCenters_x(centerOfMass_x_vec.data(), centerOfMass_x_vec.size()),
        massCenters_y(centerOfMass_y_vec.data(), centerOfMass_y_vec.size()),
        massCenters_z(centerOfMass_z_vec.data(), centerOfMass_z_vec.size()) {}

void BarnesHutOctree::computeMinMaxValuesAABB(queue &queue, buffer<double> &current_positions_x,
                                                                          buffer<double> &current_positions_y,
                                                                          buffer<double> &current_positions_z) {
    min_x = std::numeric_limits<double>::infinity();
    min_y = std::numeric_limits<double>::infinity();
    min_z = std::numeric_limits<double>::infinity();
    max_x = std::numeric_limits<double>::lowest();
    max_y = std::numeric_limits<double>::lowest();
    max_z = std::numeric_limits<double>::lowest();

    int threadCount = configuration::barnes_hut_algorithm::AABBWorkItemCount;
    int bodiesPerThread = std::ceil(numberOfBodies / threadCount);

    std::vector<double> localMax_X_vec(threadCount);
    std::vector<double> localMax_Y_vec(threadCount);
    std::vector<double> localMax_Z_vec(threadCount);

    buffer<double> localMax_X(localMax_X_vec.data(), localMax_X_vec.size());
    buffer<double> localMax_Y(localMax_Y_vec.data(), localMax_Y_vec.size());
    buffer<double> localMax_Z(localMax_Z_vec.data(), localMax_Z_vec.size());

    std::vector<double> localMin_X_vec(threadCount);
    std::vector<double> localMin_Y_vec(threadCount);
    std::vector<double> localMin_Z_vec(threadCount);

    buffer<double> localMin_X(localMin_X_vec.data(), localMin_X_vec.size());
    buffer<double> localMin_Y(localMin_Y_vec.data(), localMin_Y_vec.size());
    buffer<double> localMin_Z(localMin_Z_vec.data(), localMin_Z_vec.size());


    queue.submit([&](handler &h) {

        accessor<double> POS_X(current_positions_x, h);
        accessor<double> POS_Y(current_positions_y, h);
        accessor<double> POS_Z(current_positions_z, h);

        accessor<double> MIN_X(localMin_X, h);
        accessor<double> MIN_Y(localMin_Y, h);
        accessor<double> MIN_Z(localMin_Z, h);

        accessor<double> MAX_X(localMax_X, h);
        accessor<double> MAX_Y(localMax_Y, h);
        accessor<double> MAX_Z(localMax_Z, h);

        std::size_t N = numberOfBodies;

        h.parallel_for(sycl::range<1>(threadCount), [=](auto &idx) {

            for (int i = bodiesPerThread * idx; i < bodiesPerThread * idx + bodiesPerThread; ++i) {
                if (i < N) {
                    double current_x = POS_X[i];
                    double current_y = POS_Y[i];
                    double current_z = POS_Z[i];

                    MIN_X[idx] = sycl::min(MIN_X[idx], current_x);
                    MIN_Y[idx] = sycl::min(MIN_Y[idx], current_y);
                    MIN_Z[idx] = sycl::min(MIN_Z[idx], current_z);

                    MAX_X[idx] = sycl::max(MAX_X[idx], current_x);
                    MAX_Y[idx] = sycl::max(MAX_Y[idx], current_y);
                    MAX_Z[idx] = sycl::max(MAX_Z[idx], current_z);


                }
            }

        });
    }).wait();

    host_accessor<double> MIN_X(localMin_X);
    host_accessor<double> MIN_Y(localMin_Y);
    host_accessor<double> MIN_Z(localMin_Z);

    host_accessor<double> MAX_X(localMax_X);
    host_accessor<double> MAX_Y(localMax_Y);
    host_accessor<double> MAX_Z(localMax_Z);

    for (int i = 0; i < threadCount; ++i) {

        double current_min_x = MIN_X[i];
        double current_min_y = MIN_Y[i];
        double current_min_z = MIN_Z[i];

        double current_max_x = MAX_X[i];
        double current_max_y = MAX_Y[i];
        double current_max_z = MAX_Z[i];

        if (current_min_x < min_x) {
            min_x = current_min_x;
        }

        if (current_min_y < min_y) {
            min_y = current_min_y;
        }

        if (current_min_z < min_z) {
            min_z = current_min_z;
        }

        if (current_max_x > max_x) {
            max_x = current_max_x;
        }

        if (current_max_y > max_y) {
            max_y = current_max_y;
        }

        if (current_max_z > max_z) {
            max_z = current_max_z;
        }

    }

    // The AABB is now a square: transformation into cube
    double x_length = std::abs(max_x - min_x);
    double y_length = std::abs(max_y - min_y);
    double z_length = std::abs(max_z - min_z);
    double maxEdgeLength = std::max(x_length, std::max(y_length, z_length));

    // Extend the lengths of the shorter edges to maxEdgeLength by adding the difference equally to each side of the edge.
    if (maxEdgeLength == x_length) {
        min_z = min_z - ((maxEdgeLength - z_length) / 2);
        min_y = min_y - ((maxEdgeLength - y_length) / 2);

        max_z = max_z + ((maxEdgeLength - z_length) / 2);
        max_y = max_y + ((maxEdgeLength - y_length) / 2);
    } else if (maxEdgeLength == y_length) {
        min_x = min_x - ((maxEdgeLength - x_length) / 2);
        min_z = min_z - ((maxEdgeLength - z_length) / 2);

        max_x = max_x + ((maxEdgeLength - x_length) / 2);
        max_z = max_z + ((maxEdgeLength - z_length) / 2);
    } else if (maxEdgeLength == z_length) {
        min_x = min_x - ((maxEdgeLength - x_length) / 2);
        min_y = min_y - ((maxEdgeLength - y_length) / 2);

        max_x = max_x + ((maxEdgeLength - x_length) / 2);
        max_y = max_y + ((maxEdgeLength - y_length) / 2);
    } else {
        throw std::invalid_argument("maxEdgeLength has to be the x,y or z length");
    }

    AABB_EdgeLength = maxEdgeLength;
}