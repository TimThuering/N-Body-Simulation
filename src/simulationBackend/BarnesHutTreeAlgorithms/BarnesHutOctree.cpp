#include "BarnesHutOctree.hpp"
#include "Configuration.hpp"
#include "TimeMeasurement.hpp"

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

        nodeIsLeaf_vec(configuration::barnes_hut_algorithm::storageSizeParameter, 1),
        nodeIsLeaf(nodeIsLeaf_vec.data(), nodeIsLeaf_vec.size()),

        nodesToProcessCenterOfMass_vec(
                configuration::barnes_hut_algorithm::storageSizeParameter),
        nodesToProcessCenterOfMass(nodesToProcessCenterOfMass_vec.data(), nodesToProcessCenterOfMass_vec.size()),

        sumMasses_vec(configuration::barnes_hut_algorithm::storageSizeParameter),
        centerOfMass_x_vec(configuration::barnes_hut_algorithm::storageSizeParameter),
        centerOfMass_y_vec(configuration::barnes_hut_algorithm::storageSizeParameter),
        centerOfMass_z_vec(configuration::barnes_hut_algorithm::storageSizeParameter),
        sumOfMasses(sumMasses_vec.data(), sumMasses_vec.size()),
        massCenters_x(centerOfMass_x_vec.data(), centerOfMass_x_vec.size()),
        massCenters_y(centerOfMass_y_vec.data(), centerOfMass_y_vec.size()),
        massCenters_z(centerOfMass_z_vec.data(), centerOfMass_z_vec.size()),
        nextFreeNodeID_vec(1),
        nextFreeNodeID(nextFreeNodeID_vec.data(), nextFreeNodeID_vec.size()),
        bodyCountNode_vec(configuration::barnes_hut_algorithm::storageSizeParameter, 0),
        bodyCountNode(bodyCountNode_vec.data(), bodyCountNode_vec.size()),
        sortedBodiesInOrder_vec(configuration::numberOfBodies, configuration::numberOfBodies),
        sortedBodiesInOrder(sortedBodiesInOrder_vec.data(), sortedBodiesInOrder_vec.size()) {
}

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
    int bodiesPerThread = std::ceil((double) numberOfBodies / (double) threadCount);

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

    d_type::int_t N = numberOfBodies;

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


        h.parallel_for(sycl::range<1>(threadCount), [=](sycl::id<1> idx) {

            for (int i = bodiesPerThread * idx.get(0); i < bodiesPerThread * idx.get(0) + bodiesPerThread; ++i) {
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

        if (i < N) {
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

void BarnesHutOctree::prepareCenterOfMass(queue &queue, buffer<double> &current_positions_x,
                                          buffer<double> &current_positions_y, buffer<double> &current_positions_z,
                                          buffer<double> &masses) {

    // get the amount of nodes of the tree
    host_accessor<d_type::int_t> NUM_NODES(nextFreeNodeID);
    d_type::int_t numberOfNodes = NUM_NODES[0];
    d_type::int_t N = configuration::numberOfBodies;

    // This kernel computes the center of mass values of all nodes which are a leaf node in the tree.
    queue.submit([&](handler &h) {
        accessor<double> SUM_MASSES(sumOfMasses, h);
        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);
        accessor<double> MASSES(masses, h);
        accessor<double> POS_X(current_positions_x, h);
        accessor<double> POS_Y(current_positions_y, h);
        accessor<double> POS_Z(current_positions_z, h);
        accessor<int> NODE_IS_LEAF(nodeIsLeaf, h);
        accessor<d_type::int_t> BODY_OF_NODE(bodyOfNode, h);
        accessor<d_type::int_t> BODY_COUNT_NODE(bodyCountNode, h);

        h.parallel_for(sycl::range<1>(numberOfNodes), [=](auto &i) {
            d_type::int_t bodyInNode = BODY_OF_NODE[i];
            if ((NODE_IS_LEAF[i] == 1) && bodyInNode != N) {
                // Node is a leaf node, and it contains a body
                CENTER_OF_MASS_X[i] = POS_X[bodyInNode] * MASSES[bodyInNode];
                CENTER_OF_MASS_Y[i] = POS_Y[bodyInNode] * MASSES[bodyInNode];
                CENTER_OF_MASS_Z[i] = POS_Z[bodyInNode] * MASSES[bodyInNode];
                SUM_MASSES[i] = MASSES[bodyInNode];
                BODY_COUNT_NODE[i] = 1;
            }
        });
    }).wait();

}

void BarnesHutOctree::computeCenterOfMass_GPU(queue &queue, buffer<double> &current_positions_x,
                                              buffer<double> &current_positions_y,
                                              buffer<double> &current_positions_z,
                                              buffer<double> &masses) {
    prepareCenterOfMass(queue, current_positions_x, current_positions_y, current_positions_z, masses);

    host_accessor<d_type::int_t> NUM_NODES(nextFreeNodeID);
    d_type::int_t numberOfNodes = NUM_NODES[0];
    d_type::int_t N = configuration::numberOfBodies;
    d_type::int_t storageSize = configuration::barnes_hut_algorithm::storageSizeParameter;

    d_type::int_t nodeCountPerWorkItem;
    if (numberOfNodes > configuration::barnes_hut_algorithm::centerOfMassWorkItemCount) {
        nodeCountPerWorkItem = std::ceil((double) numberOfNodes /
                                         (double) configuration::barnes_hut_algorithm::centerOfMassWorkItemCount);
    } else {
        nodeCountPerWorkItem = 1;
    }

    d_type::int_t workItemCount = configuration::barnes_hut_algorithm::centerOfMassWorkItemCount;

    queue.submit([&](handler &h) {
        accessor<double> SUM_MASSES(sumOfMasses, h);
        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);
        accessor<int> NODE_IS_LEAF(nodeIsLeaf, h);
        accessor<d_type::int_t> OCTANTS(octants, h);
        accessor<d_type::int_t> BODY_OF_NODE(bodyOfNode, h);
        accessor<d_type::int_t> NODES_TO_PROCESS(nodesToProcessCenterOfMass, h);
        accessor<d_type::int_t> BODY_COUNT_NODE(bodyCountNode, h);

        h.parallel_for(nd_range<1>(range<1>(workItemCount), range<1>(workItemCount)), [=](auto &nd_item) {
            d_type::int_t nextID = nd_item.get_global_id() * nodeCountPerWorkItem; // start index for cached nodes

            for (d_type::int_t i = nd_item.get_global_id(); i < numberOfNodes; i += workItemCount) {

                //start with the last nodes which were created
                d_type::int_t index = numberOfNodes - 1 - i;

                d_type::int_t bodyInNode = BODY_OF_NODE[index];

                if (SUM_MASSES[index] != 0) {
                    // node already processed
                    continue;
                }

                if ((NODE_IS_LEAF[index] == 1) && bodyInNode == N) {
                    // empty node
                    continue;
                }

                if (NODE_IS_LEAF[index] == 0) {

                    bool allReady = true; // check if all children of the node are already processed
                    for (int octant = 0; octant < 8; ++octant) {
                        d_type::int_t octantID = OCTANTS[index + octant * storageSize];

                        if (!((SUM_MASSES[octantID] != 0) ||
                              ((NODE_IS_LEAF[octantID] == 1) && BODY_OF_NODE[octantID] == N))) {
                            // child node is not ready yet
                            allReady = false;
                            break;
                        }
                    }

                    if (allReady) {
                        // all child nodes are ready --> compute and store center of mass
                        double sumMasses = 0;
                        double centerMassX = 0;
                        double centerMassY = 0;
                        double centerMassZ = 0;
                        d_type::int_t bodyCount = 0;
                        for (int octant = 0; octant < 8; ++octant) {
                            d_type::int_t octantID = OCTANTS[index + octant * storageSize];
                            centerMassX += CENTER_OF_MASS_X[octantID];
                            centerMassY += CENTER_OF_MASS_Y[octantID];
                            centerMassZ += CENTER_OF_MASS_Z[octantID];
                            sumMasses += SUM_MASSES[octantID];
                            bodyCount += BODY_COUNT_NODE[octantID];
                        }
                        BODY_COUNT_NODE[index] = bodyCount;
                        CENTER_OF_MASS_X[index] = centerMassX;
                        CENTER_OF_MASS_Y[index] = centerMassY;
                        CENTER_OF_MASS_Z[index] = centerMassZ;
                        nd_item.mem_fence(access::fence_space::global_and_local);
                        SUM_MASSES[index] = sumMasses; // the sum masses values acts as a flag that this node is processed completely
                    } else {
                        // add node to nodes which have to processed.
                        NODES_TO_PROCESS[nextID] = index;
                        nextID += 1;
                    }
                }
            }

            // iterate over all nodes that have not been processed yet
            bool allNodesProcessed = false;
            while (!allNodesProcessed) {
                allNodesProcessed = true;
                for (d_type::int_t i = nd_item.get_global_id() * nodeCountPerWorkItem; i < nextID; ++i) {
                    d_type::int_t index = NODES_TO_PROCESS[i];

                    d_type::int_t bodyInNode = BODY_OF_NODE[index];

                    if (SUM_MASSES[index] != 0) {
                        // node already processed
                        continue;
                    }

                    if ((NODE_IS_LEAF[index] == 1) && bodyInNode == N) {
                        // empty node
                        continue;
                    }

                    if (NODE_IS_LEAF[index] == 0) {
                        bool allReady = true;
                        for (int octant = 0; octant < 8; ++octant) {
                            d_type::int_t octantID = OCTANTS[index + octant * storageSize];

                            if (!((SUM_MASSES[octantID] != 0) ||
                                  ((NODE_IS_LEAF[octantID] == 1) && BODY_OF_NODE[octantID] == N))) {
                                allReady = false;
                                break;
                            }
                        }

                        if (allReady) {
                            double sumMasses = 0;
                            double centerMassX = 0;
                            double centerMassY = 0;
                            double centerMassZ = 0;
                            d_type::int_t bodyCount = 0;
                            for (int octant = 0; octant < 8; ++octant) {
                                d_type::int_t octantID = OCTANTS[index + octant * storageSize];
                                centerMassX += CENTER_OF_MASS_X[octantID];
                                centerMassY += CENTER_OF_MASS_Y[octantID];
                                centerMassZ += CENTER_OF_MASS_Z[octantID];
                                sumMasses += SUM_MASSES[octantID];
                                bodyCount += BODY_COUNT_NODE[octantID];
                            }
                            BODY_COUNT_NODE[index] = bodyCount;
                            CENTER_OF_MASS_X[index] = centerMassX;
                            CENTER_OF_MASS_Y[index] = centerMassY;
                            CENTER_OF_MASS_Z[index] = centerMassZ;
                            nd_item.mem_fence(access::fence_space::global_and_local);
                            SUM_MASSES[index] = sumMasses;

                        } else {
                            allNodesProcessed = false;
                        }
                    }
                }
            }
        });
    }).wait();


}


void BarnesHutOctree::computeCenterOfMass_CPU(queue &queue, buffer<double> &current_positions_x,
                                              buffer<double> &current_positions_y, buffer<double> &current_positions_z,
                                              buffer<double> &masses) {
    prepareCenterOfMass(queue, current_positions_x, current_positions_y, current_positions_z, masses);

    host_accessor<d_type::int_t> NUM_NODES(nextFreeNodeID);
    d_type::int_t numberOfNodes = NUM_NODES[0];
    d_type::int_t N = configuration::numberOfBodies;
    d_type::int_t storageSize = configuration::barnes_hut_algorithm::storageSizeParameter;

    d_type::int_t nodeCountPerWorkItem;
    if (numberOfNodes > configuration::barnes_hut_algorithm::centerOfMassWorkItemCount) {
        nodeCountPerWorkItem = std::ceil((double) numberOfNodes /
                                         (double) configuration::barnes_hut_algorithm::centerOfMassWorkItemCount);
    } else {
        nodeCountPerWorkItem = 1;
    }

    d_type::int_t workItemCount = configuration::barnes_hut_algorithm::centerOfMassWorkItemCount;


    queue.submit([&](handler &h) {
        accessor<double> SUM_MASSES(sumOfMasses, h);
        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);
        accessor<int> NODE_IS_LEAF(nodeIsLeaf, h);
        accessor<d_type::int_t> OCTANTS(octants, h);
        accessor<d_type::int_t> BODY_OF_NODE(bodyOfNode, h);
        accessor<d_type::int_t> NODES_TO_PROCESS(nodesToProcessCenterOfMass, h);
        accessor<d_type::int_t> BODY_COUNT_NODE(bodyCountNode, h);


        h.parallel_for(nd_range<1>(range<1>(workItemCount), range<1>(workItemCount)), [=](auto &nd_item) {
            d_type::int_t nextID = nd_item.get_global_id() * nodeCountPerWorkItem;

            for (int i = 0; i < nodeCountPerWorkItem; ++i) {
                if (nd_item.get_global_id() * nodeCountPerWorkItem + i > numberOfNodes - 1) {
                    break;
                }

                // get next node assigned to this work-item
                d_type::int_t index = numberOfNodes - 1 - nd_item.get_global_id() * nodeCountPerWorkItem - i;
                d_type::int_t bodyInNode = BODY_OF_NODE[index];

                if (SUM_MASSES[index] != 0) {
                    // node already processed
                    continue;
                }

                if ((NODE_IS_LEAF[index] == 1) && bodyInNode == N) {
                    // empty node
                    continue;
                }

                if (NODE_IS_LEAF[index] == 0) {
                    bool allReady = true;
                    for (int octant = 0; octant < 8; ++octant) {
                        d_type::int_t octantID = OCTANTS[index + octant * storageSize];

                        if (!((SUM_MASSES[octantID] != 0) ||
                              ((NODE_IS_LEAF[octantID] == 1) && BODY_OF_NODE[octantID] == N))) {
                            allReady = false;
                            break;
                        }
                    }

                    if (allReady) {
                        double sumMasses = 0;
                        double centerMassX = 0;
                        double centerMassY = 0;
                        double centerMassZ = 0;
                        d_type::int_t bodyCount = 0;
                        for (int octant = 0; octant < 8; ++octant) {
                            d_type::int_t octantID = OCTANTS[index + octant * storageSize];
                            centerMassX += CENTER_OF_MASS_X[octantID];
                            centerMassY += CENTER_OF_MASS_Y[octantID];
                            centerMassZ += CENTER_OF_MASS_Z[octantID];
                            sumMasses += SUM_MASSES[octantID];
                            bodyCount += BODY_COUNT_NODE[octantID];
                        }
                        BODY_COUNT_NODE[index] = bodyCount;
                        CENTER_OF_MASS_X[index] = centerMassX;
                        CENTER_OF_MASS_Y[index] = centerMassY;
                        CENTER_OF_MASS_Z[index] = centerMassZ;
                        nd_item.mem_fence(access::fence_space::global_and_local);
                        SUM_MASSES[index] = sumMasses;

                    } else {
                        // add node to nodes which have to processed.
                        NODES_TO_PROCESS[nextID] = index;
                        nextID += 1;
                    }
                }
            }

            bool allNodesProcessed = false;
            // iterate over all nodes that still have to be processed
            while (!allNodesProcessed) {
                allNodesProcessed = true;
                for (d_type::int_t i = nd_item.get_global_id() * nodeCountPerWorkItem; i < nextID; ++i) {
                    d_type::int_t index = NODES_TO_PROCESS[i];

                    d_type::int_t bodyInNode = BODY_OF_NODE[index];

                    if (SUM_MASSES[index] != 0) {
                        // node already processed
                        continue;
                    }

                    if ((NODE_IS_LEAF[index] == 1) && bodyInNode == N) {
                        // empty node
                        continue;
                    }

                    if (NODE_IS_LEAF[index] == 0) {
                        bool allReady = true;
                        for (int octant = 0; octant < 8; ++octant) {
                            d_type::int_t octantID = OCTANTS[index + octant * storageSize];

                            if (!((SUM_MASSES[octantID] != 0) ||
                                  ((NODE_IS_LEAF[octantID] == 1) && BODY_OF_NODE[octantID] == N))) {
                                allReady = false;
                                break;
                            }
                        }

                        if (allReady) {
                            double sumMasses = 0;
                            double centerMassX = 0;
                            double centerMassY = 0;
                            double centerMassZ = 0;
                            d_type::int_t bodyCount = 0;
                            for (int octant = 0; octant < 8; ++octant) {
                                d_type::int_t octantID = OCTANTS[index + octant * storageSize];
                                centerMassX += CENTER_OF_MASS_X[octantID];
                                centerMassY += CENTER_OF_MASS_Y[octantID];
                                centerMassZ += CENTER_OF_MASS_Z[octantID];
                                sumMasses += SUM_MASSES[octantID];
                                bodyCount += BODY_COUNT_NODE[octantID];
                            }
                            BODY_COUNT_NODE[index] = bodyCount;
                            CENTER_OF_MASS_X[index] = centerMassX;
                            CENTER_OF_MASS_Y[index] = centerMassY;
                            CENTER_OF_MASS_Z[index] = centerMassZ;
                            nd_item.mem_fence(access::fence_space::global_and_local);
                            SUM_MASSES[index] = sumMasses;

                        } else {
                            allNodesProcessed = false;
                        }
                    }
                }
            }
        });
    }).wait();

}

void BarnesHutOctree::sortBodies(queue &queue, buffer<double> &current_positions_x, buffer<double> &current_positions_y,
                                 buffer<double> &current_positions_z) {

    d_type::int_t storageSize = configuration::barnes_hut_algorithm::storageSizeParameter;

    queue.submit([&](handler &h) {
        accessor<int> NODE_IS_LEAF(nodeIsLeaf, h);
        accessor<double> MIN_X(min_x_values, h);
        accessor<double> MIN_Y(min_y_values, h);
        accessor<double> MIN_Z(min_z_values, h);
        accessor<d_type::int_t> OCTANTS(octants, h);
        accessor<d_type::int_t> BODY_COUNT_NODE(bodyCountNode, h);
        accessor<double> EDGE_LENGTHS(edgeLengths, h);
        accessor<double> POS_X(current_positions_x, h);
        accessor<double> POS_Y(current_positions_y, h);
        accessor<double> POS_Z(current_positions_z, h);
        accessor<d_type::int_t> SORTED_BODIES(sortedBodiesInOrder, h);


        h.parallel_for(sycl::range<1>(configuration::numberOfBodies), [=](sycl::id<1> i) {
            d_type::int_t insertionIndex = 0;
            d_type::int_t currentNode = 0;
            bool indexDetermined = false;

            while (!indexDetermined) {
                if (NODE_IS_LEAF[currentNode]) {
                    // bottom of the tree is reached, insertion index has been determined
                    indexDetermined = true;
                } else {
                    // step down in the tree and count the amount of bodies that have to be inserted "left" of this body
                    d_type::int_t octantID;
                    double parentMin_x = MIN_X[currentNode];
                    double parentMin_y = MIN_Y[currentNode];
                    double parentMin_z = MIN_Z[currentNode];
                    double parentEdgeLength = EDGE_LENGTHS[currentNode];

                    bool upperPart = POS_Y[i] > parentMin_y + (parentEdgeLength / 2);
                    bool rightPart = POS_X[i] > parentMin_x + (parentEdgeLength / 2);
                    bool backPart = POS_Z[i] < parentMin_z + (parentEdgeLength / 2);

                    // interpret as binary and convert into decimal. octantIndex = 0 would correspond to the lowerSW node octantIndex = 7 would be the upperNE node
                    d_type::int_t octantIndex = ((int) upperPart) * 4 + ((int) rightPart) * 2 + ((int) backPart) * 1;

                    // find start index of octant type
                    d_type::int_t octantAddress = octantIndex * storageSize;

                    // get octant of current Node
                    octantAddress = octantAddress + currentNode;

                    octantID = OCTANTS[octantAddress];


                    for (d_type::int_t j = 0; j < octantIndex; ++j) {
                        // all bodies in the octants "left" to the octant of this body will get stored "left" of the current body
                        insertionIndex += BODY_COUNT_NODE[OCTANTS[j * storageSize + currentNode]];
                    }
                    currentNode = octantID;
                }
            }
            SORTED_BODIES[insertionIndex] = i.get(0);
        });
    }).wait();

}
