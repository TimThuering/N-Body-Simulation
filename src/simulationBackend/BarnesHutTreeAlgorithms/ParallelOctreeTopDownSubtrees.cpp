#include "ParallelOctreeTopDownSubtrees.hpp"
#include "Configuration.hpp"
#include "Definitions.hpp"

ParallelOctreeTopDownSubtrees::ParallelOctreeTopDownSubtrees() :
        BarnesHutOctree(),
        nodeIsLocked_vec(configuration::barnes_hut_algorithm::storageSizeParameter, 0),
        nodeIsLocked(nodeIsLocked_vec.data(), nodeIsLocked_vec.size()),
        subtreeOfBody_vec(numberOfBodies, 0),
        subtreeOfBody(subtreeOfBody_vec.data(), subtreeOfBody_vec.size()),
        subtreeOfNode_vec(configuration::barnes_hut_algorithm::storageSizeParameter, 0),
        subtreeOfNode(subtreeOfNode_vec.data(), subtreeOfNode_vec.size()),
        sortedBodies_vec(configuration::numberOfBodies),
        sortedBodies(sortedBodies_vec.data(), sortedBodies_vec.size()),
        subtreeCount_vec(1, 0),
        subtreeCount(subtreeCount_vec.data(), subtreeCount_vec.size()) {}

void ParallelOctreeTopDownSubtrees::buildOctree(queue &queue, buffer<double> &current_positions_x,
                                                buffer<double> &current_positions_y,
                                                buffer<double> &current_positions_z, buffer<double> &masses, TimeMeasurement &timer) {

    auto begin = std::chrono::steady_clock::now();
    // compute the axis aligned bounding box around all bodies
    computeMinMaxValuesAABB(queue, current_positions_x, current_positions_y, current_positions_z);
    auto endAABB_creation = std::chrono::steady_clock::now();


    buildOctreeToLevel(queue, current_positions_x, current_positions_y, current_positions_z, masses);
    auto endBuildOctreeToLevel = std::chrono::steady_clock::now();

    std::size_t nodeCount;
    {
        host_accessor<std::size_t> NodeCount(nextFreeNodeID);
        nodeCount = NodeCount[0];
        nodeCountTopOfTree = nodeCount;
    }
    std::vector<std::size_t> bodyCountSubtree_vec(nodeCount, 0);
    buffer<std::size_t> bodyCountSubtree(bodyCountSubtree_vec.data(), bodyCountSubtree_vec.size());

    std::vector<std::size_t> subtrees_vec(nodeCount);
    buffer<std::size_t> subtrees(subtrees_vec.data(), subtrees_vec.size());

    prepareSubtrees(queue, bodyCountSubtree, subtrees, nodeCount);
    auto endPrepareSubtrees = std::chrono::steady_clock::now();


    host_accessor SUBTREE_COUNT(subtreeCount);
    numberOfSubtrees = SUBTREE_COUNT[0];

    std::vector<std::size_t> bodiesOfSubtreeStartIndex_vec(numberOfSubtrees, 0);
    buffer<std::size_t> bodiesOfSubtreeStartIndex(bodiesOfSubtreeStartIndex_vec.data(),
                                                  bodiesOfSubtreeStartIndex_vec.size());

    sortBodiesForSubtrees(queue, bodyCountSubtree, subtrees, bodiesOfSubtreeStartIndex);
    auto endSortBodiesForSubtrees = std::chrono::steady_clock::now();


    buildSubtrees(queue, current_positions_x, current_positions_y,
                  current_positions_z, masses, bodiesOfSubtreeStartIndex, bodyCountSubtree, subtrees);
    auto endBuildSubtrees = std::chrono::steady_clock::now();


    computeCenterOfMass_GPU(queue, current_positions_x, current_positions_y, current_positions_z, masses);
//    computeCenterOfMassSubtrees_GPU(queue, current_positions_x, current_positions_y, current_positions_z, masses, bodyCountSubtree, subtrees);
    auto end = std::chrono::steady_clock::now();

    std::cout << "---------------------------------------------------------- "
              << std::chrono::duration<double, std::milli>(end - begin).count()
              << std::endl;


    timer.addTimeToSequence("AABB creation", std::chrono::duration<double, std::milli>(endAABB_creation - begin).count());
    timer.addTimeToSequence("Build octree to level", std::chrono::duration<double, std::milli>(endBuildOctreeToLevel - endAABB_creation).count());
    timer.addTimeToSequence("Prepare subtrees", std::chrono::duration<double, std::milli>(endPrepareSubtrees - endBuildOctreeToLevel).count());
    timer.addTimeToSequence("Sort bodies for subtrees", std::chrono::duration<double, std::milli>(endSortBodiesForSubtrees - endPrepareSubtrees).count());
    timer.addTimeToSequence("Build subtrees", std::chrono::duration<double, std::milli>(endBuildSubtrees - endSortBodiesForSubtrees).count());
    timer.addTimeToSequence("Compute center of mass", std::chrono::duration<double, std::milli>(end - endBuildSubtrees).count());

//    host_accessor MASSES(sumOfMasses);
//    host_accessor CX(massCenters_x);
//    host_accessor CY(massCenters_y);
//    host_accessor CZ(massCenters_z);
//    std::cout << MASSES[0] << std::endl;
//    std::cout << CX[0] << std::endl;
//    std::cout << CY[0] << std::endl;
//    std::cout << CZ[0] << std::endl;


}

void ParallelOctreeTopDownSubtrees::buildOctreeToLevel(queue &queue, buffer<double> &current_positions_x,
                                                       buffer<double> &current_positions_y,
                                                       buffer<double> &current_positions_z, buffer<double> &masses) {

    std::size_t N = configuration::numberOfBodies;
    std::size_t storageSize = configuration::barnes_hut_algorithm::storageSizeParameter;
    int maxLevel = configuration::barnes_hut_algorithm::maxBuildLevel;

    // set memory order for load and store operations depending on the SYCL implementation to allow compatibility with DPC++ and OpenSYCL
#ifdef USE_OPEN_SYCL
    const sycl::memory_order memoryOrderReference = sycl::memory_order::acq_rel;
    const sycl::memory_order memoryOrderLoad = sycl::memory_order::acq_rel;
    const sycl::memory_order memoryOrderStore = sycl::memory_order::acq_rel;
#else
    const sycl::memory_order memoryOrderReference = sycl::memory_order::acq_rel;
    const sycl::memory_order memoryOrderLoad = sycl::memory_order::acquire;
    const sycl::memory_order memoryOrderStore = sycl::memory_order::release;
#endif

    // initialize data structures for an empty tree
    queue.submit([&](handler &h) {

        accessor<std::size_t> OCTANTS(octants, h);
        accessor<double> EDGE_LENGTHS(edgeLengths, h);
        accessor<double> MIN_X(min_x_values, h);
        accessor<double> MIN_Y(min_y_values, h);
        accessor<double> MIN_Z(min_z_values, h);
        accessor<std::size_t> NEXT_FREE_NODE_ID(nextFreeNodeID, h);
        accessor<int> NODE_LOCKED(nodeIsLocked, h);
        accessor<int> NODE_IS_LEAF(nodeIsLeaf, h);
        accessor<double> SUM_MASSES(sumOfMasses, h);
        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);
        accessor<std::size_t> BODY_OF_NODE(bodyOfNode, h);

        double edgeLength = AABB_EdgeLength;
        double minX = min_x;
        double minY = min_y;
        double minZ = min_z;

        h.single_task([=]() {
            // root node 0: the AABB of all bodies
            EDGE_LENGTHS[0] = edgeLength;
            MIN_X[0] = minX;
            MIN_Y[0] = minY;
            MIN_Z[0] = minZ;

            // root has no children yet.
            OCTANTS[5 * storageSize] = 0;
            OCTANTS[7 * storageSize] = 0;
            OCTANTS[4 * storageSize] = 0;
            OCTANTS[6 * storageSize] = 0;
            OCTANTS[1 * storageSize] = 0;
            OCTANTS[3 * storageSize] = 0;
            OCTANTS[0] = 0;
            OCTANTS[2 * storageSize] = 0;

            NEXT_FREE_NODE_ID[0] = 1;

            NODE_LOCKED[0] = 0;
            NODE_IS_LEAF[0] = 1;


            SUM_MASSES[0] = 0;
            CENTER_OF_MASS_X[0] = 0;
            CENTER_OF_MASS_Y[0] = 0;
            CENTER_OF_MASS_Z[0] = 0;

            BODY_OF_NODE[0] = N;
        });

    }).wait();


    // build octree in parallel
    queue.submit([&](handler &h) {
        accessor<int> NODE_LOCKED(nodeIsLocked, h);
        accessor<int> NODE_IS_LEAF(nodeIsLeaf, h);
        accessor<double> POS_X(current_positions_x, h);
        accessor<double> POS_Y(current_positions_y, h);
        accessor<double> POS_Z(current_positions_z, h);
        accessor<std::size_t> OCTANTS(octants, h);
        accessor<double> EDGE_LENGTHS(edgeLengths, h);
        accessor<double> MIN_X(min_x_values, h);
        accessor<double> MIN_Y(min_y_values, h);
        accessor<double> MIN_Z(min_z_values, h);
        accessor<std::size_t> BODY_OF_NODE(bodyOfNode, h);
        accessor<double> SUM_MASSES(sumOfMasses, h);
        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);
        accessor<std::size_t> NEXT_FREE_NODE_ID(nextFreeNodeID, h);
        accessor<std::size_t> SUBTREE_OF_BODY(subtreeOfBody, h);


        // determine the maximum body count per work-item
        std::size_t bodyCountPerWorkItem;
        if (configuration::numberOfBodies > configuration::barnes_hut_algorithm::octreeWorkItemCount) {
            bodyCountPerWorkItem = std::ceil(
                    (double) configuration::numberOfBodies /
                    (double) configuration::barnes_hut_algorithm::octreeWorkItemCount);
        } else {
            bodyCountPerWorkItem = 1;
        }

        h.parallel_for(
                nd_range<1>(range<1>(configuration::barnes_hut_algorithm::octreeWorkItemCount),
                            range<1>(configuration::barnes_hut_algorithm::octreeWorkItemCount)),
                [=](auto &nd_item) {

                    atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                            access::address_space::global_space> nextFreeNodeIDAccessor(NEXT_FREE_NODE_ID[0]);


                    // for all bodies assigned to this work-item.
                    for (std::size_t i = nd_item.get_global_id() * bodyCountPerWorkItem;
                         i <
                         (std::size_t) (nd_item.get_global_id() * bodyCountPerWorkItem + bodyCountPerWorkItem); ++i) {
                        // check if body ID actually exists
                        if (i < N) {
                            SUBTREE_OF_BODY[i] = 0;
                            //start inserting at the root node
                            std::size_t currentDepth = 0;
                            std::size_t currentNode = 0;

                            bool nodeInserted = false;

                            while (!nodeInserted) {
                                // We check if the current node is locked by some other thread. If it is not, this thread locks it and
                                // continues with the insertion process.
                                int exp = 0;
                                atomic_ref<int, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicNodeIsLockedAccessor(
                                        NODE_LOCKED[currentNode]);

                                atomic_ref<int, memoryOrderReference, memory_scope::device,
                                        access::address_space::global_space> atomicNodeIsLeafAccessor(
                                        NODE_IS_LEAF[currentNode]);

                                if (currentDepth < maxLevel) {

                                    if (atomicNodeIsLeafAccessor.load(memoryOrderLoad, memory_scope::device) ==
                                        1) {
                                        // the current node is a leaf node, try to lock the node and insert body

                                        if (atomicNodeIsLockedAccessor.compare_exchange_strong(exp, 1,
                                                                                               memory_order::acq_rel,
                                                                                               memory_scope::device)) {

                                            if (atomicNodeIsLeafAccessor.load(memoryOrderLoad,
                                                                              memory_scope::device) == 1) {
                                                // node is locked and still a leaf node --> it is safe to continue with insertion

                                                if (BODY_OF_NODE[currentNode] == N) {
                                                    // the leaf node is empty --> Insert the current body into the current node and set the flag to continue with next body

                                                    BODY_OF_NODE[currentNode] = i;

                                                    nodeInserted = true;
//                                                    sycl::atomic_fence(memory_order::acq_rel, memory_scope::device);
                                                    nd_item.mem_fence(access::fence_space::global_and_local);
                                                } else {
                                                    // the leaf node already contains a body --> split the node and insert old body

                                                    std::size_t bodyIDinNode = BODY_OF_NODE[currentNode];

                                                    // determine insertion index for the new child nodes and reserve 8 indices
                                                    std::size_t incr = 8;
                                                    std::size_t firstIndex = nextFreeNodeIDAccessor.fetch_add(incr);


                                                    double childEdgeLength = EDGE_LENGTHS[currentNode] / 2;

                                                    double parent_min_x = MIN_X[currentNode];
                                                    double parent_min_y = MIN_Y[currentNode];
                                                    double parent_min_z = MIN_Z[currentNode];


                                                    // set the edge lengths of the child nodes
                                                    for (std::size_t idx = firstIndex; idx < firstIndex + 8; ++idx) {
                                                        EDGE_LENGTHS[idx] = childEdgeLength;
                                                    }

                                                    // create the 8 new octants
                                                    OCTANTS[5 * storageSize + currentNode] = firstIndex;
                                                    OCTANTS[7 * storageSize + currentNode] = firstIndex + 1;
                                                    OCTANTS[4 * storageSize + currentNode] = firstIndex + 2;
                                                    OCTANTS[6 * storageSize + currentNode] = firstIndex + 3;

                                                    OCTANTS[1 * storageSize + currentNode] = firstIndex + 4;
                                                    OCTANTS[3 * storageSize + currentNode] = firstIndex + 5;
                                                    OCTANTS[0 + currentNode] = firstIndex + 6;
                                                    OCTANTS[2 * storageSize + currentNode] = firstIndex + 7;


                                                    // min x,y,z values of the upperNW child node
                                                    MIN_X[firstIndex] = parent_min_x;
                                                    MIN_Y[firstIndex] = parent_min_y + childEdgeLength;
                                                    MIN_Z[firstIndex] = parent_min_z;

                                                    // min x,y,z values of the upperNE child node
                                                    MIN_X[firstIndex + 1] = parent_min_x + childEdgeLength;
                                                    MIN_Y[firstIndex + 1] = parent_min_y + childEdgeLength;
                                                    MIN_Z[firstIndex + 1] = parent_min_z;

                                                    // min x,y,z values of the upperSW child node
                                                    MIN_X[firstIndex + 2] = parent_min_x;
                                                    MIN_Y[firstIndex + 2] = parent_min_y + childEdgeLength;
                                                    MIN_Z[firstIndex + 2] = parent_min_z + childEdgeLength;

                                                    // min x,y,z values of the upperSE child node
                                                    MIN_X[firstIndex + 3] = parent_min_x + childEdgeLength;
                                                    MIN_Y[firstIndex + 3] = parent_min_y + childEdgeLength;
                                                    MIN_Z[firstIndex + 3] = parent_min_z + childEdgeLength;

                                                    // min x,y,z values of the lowerNW child node
                                                    MIN_X[firstIndex + 4] = parent_min_x;
                                                    MIN_Y[firstIndex + 4] = parent_min_y;
                                                    MIN_Z[firstIndex + 4] = parent_min_z;

                                                    // min x,y,z values of the lowerNE child node
                                                    MIN_X[firstIndex + 5] = parent_min_x + childEdgeLength;
                                                    MIN_Y[firstIndex + 5] = parent_min_y;
                                                    MIN_Z[firstIndex + 5] = parent_min_z;

                                                    // min x,y,z values of the lowerSW child node
                                                    MIN_X[firstIndex + 6] = parent_min_x;
                                                    MIN_Y[firstIndex + 6] = parent_min_y;
                                                    MIN_Z[firstIndex + 6] = parent_min_z + childEdgeLength;

                                                    // min x,y,z values of the lowerSE child node
                                                    MIN_X[firstIndex + 7] = parent_min_x + childEdgeLength;
                                                    MIN_Y[firstIndex + 7] = parent_min_y;
                                                    MIN_Z[firstIndex + 7] = parent_min_z + childEdgeLength;

                                                    // initially, the newly created octants will not have any children.
                                                    // 0 is also the root node index, but since the root will never be a child of any node, it can be used here to identify
                                                    // leaf nodes.
                                                    // Furthermore, since these nodes do not contain any bodies yet, the impossible body ID numberOfBodies gets used.
                                                    for (std::size_t idx = firstIndex; idx < firstIndex + 8; ++idx) {
                                                        OCTANTS[5 * storageSize + idx] = 0;
                                                        OCTANTS[7 * storageSize + idx] = 0;
                                                        OCTANTS[4 * storageSize + idx] = 0;
                                                        OCTANTS[6 * storageSize + idx] = 0;
                                                        OCTANTS[1 * storageSize + idx] = 0;
                                                        OCTANTS[3 * storageSize + idx] = 0;
                                                        OCTANTS[0 + idx] = 0;
                                                        OCTANTS[2 * storageSize + idx] = 0;

                                                        BODY_OF_NODE[idx] = N;

                                                        NODE_IS_LEAF[idx] = 1;

                                                        CENTER_OF_MASS_X[idx] = 0;
                                                        CENTER_OF_MASS_Y[idx] = 0;
                                                        CENTER_OF_MASS_Z[idx] = 0;

                                                        SUM_MASSES[idx] = 0;
                                                    }

                                                    // determine the new octant for the old body
                                                    std::size_t octantID;
                                                    double parentMin_x = MIN_X[currentNode];
                                                    double parentMin_y = MIN_Y[currentNode];
                                                    double parentMin_z = MIN_Z[currentNode];
                                                    double parentEdgeLength = EDGE_LENGTHS[currentNode];

                                                    bool upperPart =
                                                            POS_Y[bodyIDinNode] >
                                                            parentMin_y + (parentEdgeLength / 2.0);
                                                    bool rightPart =
                                                            POS_X[bodyIDinNode] >
                                                            parentMin_x + (parentEdgeLength / 2.0);
                                                    bool backPart =
                                                            POS_Z[bodyIDinNode] <
                                                            parentMin_z + (parentEdgeLength / 2.0);

                                                    // interpret as binary and convert into decimal
                                                    std::size_t octantAddress =
                                                            ((int) upperPart) * 4 + ((int) rightPart) * 2 +
                                                            ((int) backPart) * 1;
                                                    // find start index of octant type
                                                    octantAddress = octantAddress * storageSize;

                                                    // get octant of current Node
                                                    octantAddress = octantAddress + currentNode;

                                                    octantID = OCTANTS[octantAddress];

                                                    // insert the old body into the new octant it belongs to and remove it from the parent node
                                                    BODY_OF_NODE[octantID] = bodyIDinNode;
                                                    BODY_OF_NODE[currentNode] = N;

                                                    if (currentDepth == (maxLevel - 1)) {
                                                        // after the split, the new octants will be on the max level.
                                                        // the body with bodyIDinNode belongs to the subtree where the root is the respective octant
                                                        BODY_OF_NODE[octantID] = N;
                                                        SUBTREE_OF_BODY[bodyIDinNode] = octantID;
                                                    }


//                                                    sycl::atomic_fence(memory_order::acq_rel, memory_scope::device);
                                                    nd_item.mem_fence(access::fence_space::global_and_local);
                                                    // mark the current node as a non leaf node
                                                    atomicNodeIsLeafAccessor.store(0, memoryOrderStore,
                                                                                   memory_scope::device);
                                                }

                                            }
                                            // release the lock
                                            nd_item.mem_fence(access::fence_space::global_and_local);
//                                            sycl::atomic_fence(memory_order::acq_rel, memory_scope::device);
                                            atomicNodeIsLockedAccessor.fetch_sub(1, memory_order::acq_rel,
                                                                                 memory_scope::device);
                                        }
                                    } else {
                                        // the current node is not a leaf node, i.e. it has 8 children
                                        // --> determine the octant, the body has to be inserted and set this octant as current node.

                                        std::size_t octantID;
                                        double parentMin_x = MIN_X[currentNode];
                                        double parentMin_y = MIN_Y[currentNode];
                                        double parentMin_z = MIN_Z[currentNode];
                                        double parentEdgeLength = EDGE_LENGTHS[currentNode];

                                        bool upperPart = POS_Y[i] > parentMin_y + (parentEdgeLength / 2.0);
                                        bool rightPart = POS_X[i] > parentMin_x + (parentEdgeLength / 2.0);
                                        bool backPart = POS_Z[i] < parentMin_z + (parentEdgeLength / 2.0);

                                        // interpret as binary and convert into decimal
                                        std::size_t octantAddress =
                                                ((int) upperPart) * 4 + ((int) rightPart) * 2 + ((int) backPart) * 1;

                                        // find start index of octant type
                                        octantAddress = octantAddress * storageSize;

                                        // get octant of current Node
                                        octantAddress = octantAddress + currentNode;

                                        octantID = OCTANTS[octantAddress];


                                        currentNode = octantID;
                                        currentDepth += 1;
                                    }

                                } else {
                                    // The maximum depth is reached. This node now represents a root of a subtree.
                                    // The node does not get split anymore in this Phase.
                                    nodeInserted = true;

                                    // The subtree of the current body i, will be this subtree
                                    SUBTREE_OF_BODY[i] = currentNode;
                                }
                            }
                        }
                    }
                });
    }).wait();

}

void ParallelOctreeTopDownSubtrees::prepareSubtrees(queue &queue, buffer<std::size_t> &bodyCountSubtree,
                                                    buffer<std::size_t> &subtrees,
                                                    std::size_t nodeCount) {

    // iterate over all bodies and determine the body count for each subtree
    queue.submit([&](handler &h) {

        accessor<std::size_t> SUBTREE_OF_BODY(subtreeOfBody, h);
        accessor<std::size_t> BODY_COUNT_SUBTREE(bodyCountSubtree, h);

        h.parallel_for(sycl::range<1>(configuration::numberOfBodies), [=](auto &i) {

            std::size_t currentSubtreeNode = SUBTREE_OF_BODY[i];
            atomic_ref<std::size_t, memory_order::relaxed, memory_scope::device,
                    access::address_space::global_space> bodyCounter(BODY_COUNT_SUBTREE[currentSubtreeNode]);
            std::size_t incr = 1;
            bodyCounter.fetch_add(incr, memory_order::relaxed, memory_scope::device);

        });
    }).wait();

    // iterate over all nodes and determine the subtrees
    queue.submit([&](handler &h) {

        accessor<std::size_t> SUBTREES(subtrees, h);
        accessor<std::size_t> BODY_COUNT_SUBTREE(bodyCountSubtree, h);
        accessor<std::size_t> SUBTREE_COUNT(subtreeCount, h);

        h.single_task([=]() {

            std::size_t nextIndex = 0;
            for (std::size_t i = 1; i < nodeCount; ++i) {

                if (BODY_COUNT_SUBTREE[i] > 0) {
                    SUBTREES[nextIndex] = i;
                    nextIndex += 1;
                }
            }
            SUBTREE_COUNT[0] = nextIndex;
        });
    }).wait();


}

void ParallelOctreeTopDownSubtrees::sortBodiesForSubtrees(queue &queue, buffer<std::size_t> &bodyCountSubtree,
                                                          buffer<std::size_t> &subtrees,
                                                          buffer<std::size_t> &bodiesOfSubtreeStartIndex) {

    std::vector<std::size_t> nextPositions_vec(numberOfSubtrees, 0);
    buffer<std::size_t> nextPositions(nextPositions_vec.data(), nextPositions_vec.size());


    queue.submit([&](handler &h) {

        accessor<std::size_t> BODY_COUNT_SUBTREE(bodyCountSubtree, h);
        accessor<std::size_t> SUBTREES(subtrees, h);
        accessor<std::size_t> START_INDEX(bodiesOfSubtreeStartIndex, h);
        accessor<std::size_t> NEXT_POSITIONS(nextPositions, h);


        h.parallel_for(sycl::range<1>(numberOfSubtrees), [=](auto &i) {
            // determine first index;
            std::size_t firstIndex = 0;
            for (int j = i - 1; j >= 0; j--) {
                firstIndex += BODY_COUNT_SUBTREE[SUBTREES[j]];
            }
            START_INDEX[i] = firstIndex;
            NEXT_POSITIONS[i] = firstIndex;

        });
    }).wait();

    queue.submit([&](handler &h) {

        accessor<std::size_t> SUBTREE_OF_BODY(subtreeOfBody, h);
        accessor<std::size_t> SORTED_BODIES(sortedBodies, h);
        accessor<std::size_t> NEXT_POSITIONS(nextPositions, h);
        accessor<std::size_t> SUBTREES(subtrees, h);
        std::size_t NUM_SUBTREES = numberOfSubtrees;


        h.parallel_for(sycl::range<1>(numberOfBodies), [=](auto &i) {

            std::size_t subtreeOfCurrentBody = SUBTREE_OF_BODY[i];


            int subtreeIndex = -1;
            for (int j = 0; j < NUM_SUBTREES; ++j) {
                if (SUBTREES[j] == subtreeOfCurrentBody) {
                    subtreeIndex = j;
                    break;
                }
            }

            if (subtreeIndex != -1) {
                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                        access::address_space::global_space> atomicNextPositionAccessor(
                        NEXT_POSITIONS[subtreeIndex]);

                std::size_t incr = 1;
                std::size_t insertionIndex = atomicNextPositionAccessor.fetch_add(incr);

                SORTED_BODIES[insertionIndex] = i;
            }


        });
    }).wait();

}


void ParallelOctreeTopDownSubtrees::buildSubtrees(queue &queue, buffer<double> &current_positions_x,
                                                  buffer<double> &current_positions_y,
                                                  buffer<double> &current_positions_z, buffer<double> &masses,
                                                  buffer<std::size_t> &bodiesOfSubtreeStartIndex,
                                                  buffer<std::size_t> &bodyCountSubtree,
                                                  buffer<std::size_t> &subtrees) {



    // storage for the maxTreeDepth which will be atomically accessed
    std::vector<std::size_t> maxTreeDepth_vec(1, 0);
    buffer<std::size_t> maxTreeDepths(maxTreeDepth_vec.data(), maxTreeDepth_vec.size());

    std::size_t N = configuration::numberOfBodies;
    std::size_t storageSize = configuration::barnes_hut_algorithm::storageSizeParameter;

    std::cout << configuration::use_OpenSYCL << std::endl;


    // set memory order for load and store operations depending on the SYCL implementation to allow compatibility with DPC++ and OpenSYCL
#ifdef USE_OPEN_SYCL
    const sycl::memory_order memoryOrderReference = sycl::memory_order::acq_rel;
    const sycl::memory_order memoryOrderLoad = sycl::memory_order::acq_rel;
    const sycl::memory_order memoryOrderStore = sycl::memory_order::acq_rel;
#else
    const sycl::memory_order memoryOrderReference = sycl::memory_order::acq_rel;
    const sycl::memory_order memoryOrderLoad = sycl::memory_order::acquire;
    const sycl::memory_order memoryOrderStore = sycl::memory_order::release;
#endif


    // build octree in parallel
    queue.submit([&](handler &h) {


        accessor<int> NODE_LOCKED(nodeIsLocked, h);
        accessor<int> NODE_IS_LEAF(nodeIsLeaf, h);
        accessor<double> POS_X(current_positions_x, h);
        accessor<double> POS_Y(current_positions_y, h);
        accessor<double> POS_Z(current_positions_z, h);
        accessor<std::size_t> OCTANTS(octants, h);
        accessor<double> EDGE_LENGTHS(edgeLengths, h);
        accessor<double> MIN_X(min_x_values, h);
        accessor<double> MIN_Y(min_y_values, h);
        accessor<double> MIN_Z(min_z_values, h);
        accessor<std::size_t> BODY_OF_NODE(bodyOfNode, h);
        accessor<double> SUM_MASSES(sumOfMasses, h);
        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);
        accessor<std::size_t> NEXT_FREE_NODE_ID(nextFreeNodeID, h);
        accessor<std::size_t> START_INDEX(bodiesOfSubtreeStartIndex, h);
        accessor<std::size_t> BODY_COUNT_SUBTREE(bodyCountSubtree, h);
        accessor<std::size_t> SORTED_BODIES(sortedBodies, h);
        accessor<std::size_t> SUBTREES(subtrees, h);
        accessor<std::size_t> SUBTREE_OF_NODE(subtreeOfNode, h);


        std::size_t maxLevel = configuration::barnes_hut_algorithm::maxBuildLevel;
        std::size_t localSize = configuration::barnes_hut_algorithm::octreeWorkItemCount;

        h.parallel_for(nd_range<1>(range<1>(numberOfSubtrees * localSize),
                                   range<1>(localSize)),
                       [=](auto &nd_item) {

                           atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                   access::address_space::global_space> nextFreeNodeIDAccessor(NEXT_FREE_NODE_ID[0]);

                           std::size_t workGroupID = nd_item.get_group_linear_id();
                           std::size_t subTreeRootNode = SUBTREES[workGroupID];
                           std::size_t bodyCountWorkGroup = BODY_COUNT_SUBTREE[subTreeRootNode];
                           std::size_t startIndexBodies = START_INDEX[workGroupID];

                           std::size_t bodyCountPerWorkItem;
                           if (bodyCountWorkGroup > localSize) {
                               bodyCountPerWorkItem = sycl::ceil((double) bodyCountWorkGroup / (double) localSize);
                           } else {
                               bodyCountPerWorkItem = 1;
                               if (nd_item.get_local_id() > bodyCountWorkGroup) {
                                   return;
                               }
                           }


                           // for all bodies assigned to this work-item.
                           for (std::size_t index = startIndexBodies + nd_item.get_local_id() * bodyCountPerWorkItem;
                                index <
                                (std::size_t) (startIndexBodies + nd_item.get_local_id() * bodyCountPerWorkItem +
                                               bodyCountPerWorkItem); ++index) {

                               // check if body ID is actually part of the body IDs of this work group
                               if (index < (startIndexBodies + bodyCountWorkGroup)) {
                                   // the id of the current body
                                   std::size_t i = SORTED_BODIES[index];

                                   //start inserting at the subtree root node
                                   std::size_t currentDepth = maxLevel;
                                   std::size_t currentNode = subTreeRootNode;

                                   bool nodeInserted = false;


                                   while (!nodeInserted) {
                                       // We check if the current node is locked by some other thread. If it is not, this thread locks it and
                                       // continues with the insertion process.
                                       int exp = 0;
                                       atomic_ref<int, memory_order::acq_rel, memory_scope::device,
                                               access::address_space::global_space> atomicNodeIsLockedAccessor(
                                               NODE_LOCKED[currentNode]);

                                       atomic_ref<int, memoryOrderReference, memory_scope::device,
                                               access::address_space::global_space> atomicNodeIsLeafAccessor(
                                               NODE_IS_LEAF[currentNode]);

                                       if (atomicNodeIsLeafAccessor.load(memoryOrderLoad,
                                                                         memory_scope::device) ==
                                           1) {
                                           // the current node is a leaf node, try to lock the node and insert body

                                           if (atomicNodeIsLockedAccessor.compare_exchange_strong(exp, 1,
                                                                                                  memory_order::acq_rel,
                                                                                                  memory_scope::device)) {

                                               if (atomicNodeIsLeafAccessor.load(memoryOrderLoad,
                                                                                 memory_scope::device) == 1) {
                                                   // node is locked and still a leaf node --> it is safe to continue with insertion

                                                   if (BODY_OF_NODE[currentNode] == N) {
                                                       // the leaf node is empty --> Insert the current body into the current node and set the flag to continue with next body

                                                       BODY_OF_NODE[currentNode] = i;

//                                                       max_tree_depth.fetch_max(currentDepth);
                                                       nodeInserted = true;
                                                       //sycl::atomic_fence(memory_order::acq_rel, memory_scope::device);
                                                       nd_item.mem_fence(access::fence_space::global_and_local);
                                                   } else {
                                                       // the leaf node already contains a body --> split the node and insert old body

                                                       std::size_t bodyIDinNode = BODY_OF_NODE[currentNode];

                                                       // determine insertion index for the new child nodes and reserve 8 indices
                                                       std::size_t incr = 8;
                                                       std::size_t firstIndex = nextFreeNodeIDAccessor.fetch_add(
                                                               incr);


                                                       double childEdgeLength = EDGE_LENGTHS[currentNode] / 2;

                                                       double parent_min_x = MIN_X[currentNode];
                                                       double parent_min_y = MIN_Y[currentNode];
                                                       double parent_min_z = MIN_Z[currentNode];


                                                       // set the edge lengths of the child nodes
                                                       for (std::size_t idx = firstIndex;
                                                            idx < firstIndex + 8; ++idx) {
                                                           EDGE_LENGTHS[idx] = childEdgeLength;
                                                       }

                                                       // create the 8 new octants
                                                       OCTANTS[5 * storageSize + currentNode] = firstIndex;
                                                       OCTANTS[7 * storageSize + currentNode] = firstIndex + 1;
                                                       OCTANTS[4 * storageSize + currentNode] = firstIndex + 2;
                                                       OCTANTS[6 * storageSize + currentNode] = firstIndex + 3;

                                                       OCTANTS[1 * storageSize + currentNode] = firstIndex + 4;
                                                       OCTANTS[3 * storageSize + currentNode] = firstIndex + 5;
                                                       OCTANTS[0 + currentNode] = firstIndex + 6;
                                                       OCTANTS[2 * storageSize + currentNode] = firstIndex + 7;


                                                       // min x,y,z values of the upperNW child node
                                                       MIN_X[firstIndex] = parent_min_x;
                                                       MIN_Y[firstIndex] = parent_min_y + childEdgeLength;
                                                       MIN_Z[firstIndex] = parent_min_z;

                                                       // min x,y,z values of the upperNE child node
                                                       MIN_X[firstIndex + 1] = parent_min_x + childEdgeLength;
                                                       MIN_Y[firstIndex + 1] = parent_min_y + childEdgeLength;
                                                       MIN_Z[firstIndex + 1] = parent_min_z;

                                                       // min x,y,z values of the upperSW child node
                                                       MIN_X[firstIndex + 2] = parent_min_x;
                                                       MIN_Y[firstIndex + 2] = parent_min_y + childEdgeLength;
                                                       MIN_Z[firstIndex + 2] = parent_min_z + childEdgeLength;

                                                       // min x,y,z values of the upperSE child node
                                                       MIN_X[firstIndex + 3] = parent_min_x + childEdgeLength;
                                                       MIN_Y[firstIndex + 3] = parent_min_y + childEdgeLength;
                                                       MIN_Z[firstIndex + 3] = parent_min_z + childEdgeLength;

                                                       // min x,y,z values of the lowerNW child node
                                                       MIN_X[firstIndex + 4] = parent_min_x;
                                                       MIN_Y[firstIndex + 4] = parent_min_y;
                                                       MIN_Z[firstIndex + 4] = parent_min_z;

                                                       // min x,y,z values of the lowerNE child node
                                                       MIN_X[firstIndex + 5] = parent_min_x + childEdgeLength;
                                                       MIN_Y[firstIndex + 5] = parent_min_y;
                                                       MIN_Z[firstIndex + 5] = parent_min_z;

                                                       // min x,y,z values of the lowerSW child node
                                                       MIN_X[firstIndex + 6] = parent_min_x;
                                                       MIN_Y[firstIndex + 6] = parent_min_y;
                                                       MIN_Z[firstIndex + 6] = parent_min_z + childEdgeLength;

                                                       // min x,y,z values of the lowerSE child node
                                                       MIN_X[firstIndex + 7] = parent_min_x + childEdgeLength;
                                                       MIN_Y[firstIndex + 7] = parent_min_y;
                                                       MIN_Z[firstIndex + 7] = parent_min_z + childEdgeLength;

                                                       // initially, the newly created octants will not have any children.
                                                       // 0 is also the root node index, but since the root will never be a child of any node, it can be used here to identify
                                                       // leaf nodes.
                                                       // Furthermore, since these nodes do not contain any bodies yet, the impossible body ID numberOfBodies gets used.
                                                       for (std::size_t idx = firstIndex;
                                                            idx < firstIndex + 8; ++idx) {
                                                           OCTANTS[5 * storageSize + idx] = 0;
                                                           OCTANTS[7 * storageSize + idx] = 0;
                                                           OCTANTS[4 * storageSize + idx] = 0;
                                                           OCTANTS[6 * storageSize + idx] = 0;
                                                           OCTANTS[1 * storageSize + idx] = 0;
                                                           OCTANTS[3 * storageSize + idx] = 0;
                                                           OCTANTS[0 + idx] = 0;
                                                           OCTANTS[2 * storageSize + idx] = 0;


                                                           BODY_OF_NODE[idx] = N;
                                                           SUBTREE_OF_NODE[idx] = subTreeRootNode;

                                                           NODE_IS_LEAF[idx] = 1;

                                                           CENTER_OF_MASS_X[idx] = 0;
                                                           CENTER_OF_MASS_Y[idx] = 0;
                                                           CENTER_OF_MASS_Z[idx] = 0;

                                                           SUM_MASSES[idx] = 0;
                                                       }

                                                       // determine the new octant for the old body
                                                       std::size_t octantID;
                                                       double parentMin_x = MIN_X[currentNode];
                                                       double parentMin_y = MIN_Y[currentNode];
                                                       double parentMin_z = MIN_Z[currentNode];
                                                       double parentEdgeLength = EDGE_LENGTHS[currentNode];

                                                       bool upperPart =
                                                               POS_Y[bodyIDinNode] >
                                                               parentMin_y + (parentEdgeLength / 2);
                                                       bool rightPart =
                                                               POS_X[bodyIDinNode] >
                                                               parentMin_x + (parentEdgeLength / 2);
                                                       bool backPart =
                                                               POS_Z[bodyIDinNode] <
                                                               parentMin_z + (parentEdgeLength / 2);

                                                       // interpret as binary and convert into decimal
                                                       std::size_t octantAddress =
                                                               ((int) upperPart) * 4 + ((int) rightPart) * 2 +
                                                               ((int) backPart) * 1;
                                                       // find start index of octant type
                                                       octantAddress = octantAddress * storageSize;

                                                       // get octant of current Node
                                                       octantAddress = octantAddress + currentNode;

                                                       octantID = OCTANTS[octantAddress];

                                                       // insert the old body into the new octant it belongs to and remove it from the parent node
                                                       BODY_OF_NODE[octantID] = bodyIDinNode;
                                                       BODY_OF_NODE[currentNode] = N;


                                                       //sycl::atomic_fence(memory_order::acq_rel, memory_scope::device);
                                                       nd_item.mem_fence(access::fence_space::global_and_local);
                                                       // mark the current node as a non leaf node
                                                       atomicNodeIsLeafAccessor.store(0, memoryOrderStore,
                                                                                      memory_scope::device);
                                                   }
                                                   // sycl::atomic_fence(memory_order::acq_rel, memory_scope::device);
                                               }
                                               // release the lock
                                               nd_item.mem_fence(access::fence_space::global_and_local);
                                               atomicNodeIsLockedAccessor.fetch_sub(1, memory_order::acq_rel,
                                                                                    memory_scope::device);
                                           }
                                       } else {
                                           // the current node is not a leaf node, i.e. it has 8 children
                                           // --> determine the octant, the body has to be inserted and set this octant as current node.

                                           std::size_t octantID;
                                           double parentMin_x = MIN_X[currentNode];
                                           double parentMin_y = MIN_Y[currentNode];
                                           double parentMin_z = MIN_Z[currentNode];
                                           double parentEdgeLength = EDGE_LENGTHS[currentNode];

                                           bool upperPart = POS_Y[i] > parentMin_y + (parentEdgeLength / 2);
                                           bool rightPart = POS_X[i] > parentMin_x + (parentEdgeLength / 2);
                                           bool backPart = POS_Z[i] < parentMin_z + (parentEdgeLength / 2);

                                           // interpret as binary and convert into decimal
                                           std::size_t octantAddress =
                                                   ((int) upperPart) * 4 + ((int) rightPart) * 2 +
                                                   ((int) backPart) * 1;

                                           // find start index of octant type
                                           octantAddress = octantAddress * storageSize;

                                           // get octant of current Node
                                           octantAddress = octantAddress + currentNode;

                                           octantID = OCTANTS[octantAddress];

                                           currentNode = octantID;
                                           currentDepth += 1;
                                       }
//                                           nd_item.barrier();
                                   }
//                                   }
                               }
                           }
                       });
    }).wait();
}

void ParallelOctreeTopDownSubtrees::computeCenterOfMassSubtrees_GPU(queue &queue, buffer<double> &current_positions_x,
                                                                    buffer<double> &current_positions_y,
                                                                    buffer<double> &current_positions_z,
                                                                    buffer<double> &masses,
                                                                    buffer<std::size_t> &bodyCountSubtree,
                                                                    buffer<std::size_t> &subtrees) {

    // get the amount of nodes of the tree
    host_accessor<std::size_t> NUM_NODES(nextFreeNodeID);
    std::size_t numberOfNodes = NUM_NODES[0];
    std::size_t N = configuration::numberOfBodies;

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
        accessor<std::size_t> BODY_OF_NODE(bodyOfNode, h);

        h.parallel_for(sycl::range<1>(numberOfNodes), [=](auto &i) {
            std::size_t bodyInNode = BODY_OF_NODE[i];
            if ((NODE_IS_LEAF[i] == 1) && bodyInNode != N) {
                // Node is a leaf node, and it contains a body
                CENTER_OF_MASS_X[i] = POS_X[bodyInNode] * MASSES[bodyInNode];
                CENTER_OF_MASS_Y[i] = POS_Y[bodyInNode] * MASSES[bodyInNode];
                CENTER_OF_MASS_Z[i] = POS_Z[bodyInNode] * MASSES[bodyInNode];
                SUM_MASSES[i] = MASSES[bodyInNode];
            }
        });
    }).wait();

    std::size_t storageSize = configuration::barnes_hut_algorithm::storageSizeParameter;


    std::size_t workItemCount = configuration::barnes_hut_algorithm::octreeWorkItemCount;


    queue.submit([&](handler &h) {
        accessor<double> SUM_MASSES(sumOfMasses, h);
        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);
        accessor<int> NODE_IS_LEAF(nodeIsLeaf, h);
        accessor<std::size_t> OCTANTS(octants, h);
        accessor<std::size_t> BODY_OF_NODE(bodyOfNode, h);
        accessor<std::size_t> SUBTREE_OF_NODE(subtreeOfNode, h);
        accessor<std::size_t> SUBTREES(subtrees, h);


        h.parallel_for(nd_range<1>(range<1>(workItemCount * numberOfSubtrees), range<1>(workItemCount)),
                       [=](auto &nd_item) {

                           bool allNodesProcessed = false;

                           std::size_t subtreeWorkGroup = SUBTREES[nd_item.get_group_linear_id()];
                           while (!allNodesProcessed) {
                               allNodesProcessed = true;
                               //start from the back so nodes that were created last (lower level) come first.

                               for (std::size_t i = nd_item.get_local_id(); i < numberOfNodes; i += workItemCount) {
                                   std::size_t index = numberOfNodes - 1 - i;

                                   if (SUBTREE_OF_NODE[index] != subtreeWorkGroup) {
                                       continue;
                                   }

                                   std::size_t bodyOfNode = BODY_OF_NODE[index];

                                   if (SUM_MASSES[index] != 0) {
                                       // node already processed
                                       continue;
                                   }
                                   if ((NODE_IS_LEAF[index] == 1) && bodyOfNode == N) {
                                       // empty node
                                       continue;
                                   }

                                   if (NODE_IS_LEAF[index] == 0) {

                                       bool allReady = true;
                                       for (int octant = 0; octant < 8; ++octant) {
                                           std::size_t octantID = OCTANTS[index + octant * storageSize];

                                           if (!((SUM_MASSES[octantID] != 0) ||
                                                 ((NODE_IS_LEAF[octantID] == 1) && BODY_OF_NODE[octantID] == N))) {
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
                                           for (int octant = 0; octant < 8; ++octant) {
                                               std::size_t octantID = OCTANTS[index + octant * storageSize];
                                               centerMassX += CENTER_OF_MASS_X[octantID];
                                               centerMassY += CENTER_OF_MASS_Y[octantID];
                                               centerMassZ += CENTER_OF_MASS_Z[octantID];
                                               sumMasses += SUM_MASSES[octantID];
                                           }
                                           CENTER_OF_MASS_X[index] = centerMassX;
                                           CENTER_OF_MASS_Y[index] = centerMassY;
                                           CENTER_OF_MASS_Z[index] = centerMassZ;
                                           nd_item.mem_fence(access::fence_space::global_and_local);
                                           SUM_MASSES[index] = sumMasses; // the sum masses values acts as a flag that this node is processed completely

                                       } else {
                                           allNodesProcessed = false;
                                       }


                                   }
                               }
                           }
                       });
    }).wait();

    numberOfNodes = nodeCountTopOfTree;

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
        accessor<std::size_t> OCTANTS(octants, h);
        accessor<std::size_t> BODY_OF_NODE(bodyOfNode, h);

        h.parallel_for(nd_range<1>(range<1>(workItemCount), range<1>(workItemCount)),
                       [=](auto &nd_item) {

                           bool allNodesProcessed = false;
                           while (!allNodesProcessed) {
                               allNodesProcessed = true;
                               //start from the back so nodes that were created last (lower level) come first.

//                               for (int i = 0; i < nodeCountPerWorkItem; ++i)  {
                               for (std::size_t i = nd_item.get_global_id(); i < numberOfNodes; i += workItemCount) {
                                   std::size_t index = numberOfNodes - 1 - i;


//                                   if (nd_item.get_global_id() * nodeCountPerWorkItem + i > numberOfNodes - 1) {
//                                       break;
//                                   }
//                                   std::size_t index = numberOfNodes - 1 - nd_item.get_global_id() * nodeCountPerWorkItem - i;

                                   std::size_t bodyOfNode = BODY_OF_NODE[index];

                                   if (SUM_MASSES[index] != 0) {
                                       // node already processed
                                       continue;
                                   }
                                   if ((NODE_IS_LEAF[index] == 1) && bodyOfNode == N) {
                                       // empty node
                                       continue;
                                   }

                                   if ((NODE_IS_LEAF[index] == 1) && bodyOfNode != N) {
                                       // Node is a leaf node, and it contains a body
                                       CENTER_OF_MASS_X[index] = POS_X[bodyOfNode] * MASSES[bodyOfNode];
                                       CENTER_OF_MASS_Y[index] = POS_Y[bodyOfNode] * MASSES[bodyOfNode];
                                       CENTER_OF_MASS_Z[index] = POS_Z[bodyOfNode] * MASSES[bodyOfNode];
                                       nd_item.mem_fence(access::fence_space::global_and_local);
                                       SUM_MASSES[index] = MASSES[bodyOfNode];

                                   } else if (NODE_IS_LEAF[index] == 0) {

                                       bool allReady = true;
                                       for (int octant = 0; octant < 8; ++octant) {
                                           std::size_t octantID = OCTANTS[index + octant * storageSize];

                                           if (!((SUM_MASSES[octantID] != 0) ||
                                                 ((NODE_IS_LEAF[octantID] == 1) && BODY_OF_NODE[octantID] == N))) {
                                               allReady = false;
                                           }
                                       }
                                       if (allReady) {

                                           double sumMasses = 0;
                                           for (int octant = 0; octant < 8; ++octant) {
                                               std::size_t octantID = OCTANTS[index + octant * storageSize];

                                               CENTER_OF_MASS_X[index] += CENTER_OF_MASS_X[octantID];
                                               CENTER_OF_MASS_Y[index] += CENTER_OF_MASS_Y[octantID];
                                               CENTER_OF_MASS_Z[index] += CENTER_OF_MASS_Z[octantID];
                                               sumMasses += SUM_MASSES[octantID];

                                           }
                                           nd_item.mem_fence(access::fence_space::global_and_local);
                                           SUM_MASSES[index] = sumMasses;

                                       } else {
                                           allNodesProcessed = false;
                                       }


                                   }

//                                   nd_item.barrier();


                               }
//                               nd_item.barrier();
                           }


                       });
    }).wait();


}




