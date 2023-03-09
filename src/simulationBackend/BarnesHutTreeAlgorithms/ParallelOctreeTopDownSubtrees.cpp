#include "ParallelOctreeTopDownSubtrees.hpp"
#include "Configuration.hpp"

ParallelOctreeTopDownSubtrees::ParallelOctreeTopDownSubtrees() :
        BarnesHutOctree(),
        nodeIsLocked_vec(configuration::barnes_hut_algorithm::storageSizeParameter, 0),
        nodeIsLocked(nodeIsLocked_vec.data(), nodeIsLocked_vec.size()),
        subtreeOfBody_vec(numberOfBodies, 0),
        subtreeOfBody(subtreeOfBody_vec.data(), subtreeOfBody_vec.size()),
        sortedBodies_vec(configuration::numberOfBodies),
        sortedBodies(sortedBodies_vec.data(), sortedBodies_vec.size()),
        subtreeCount_vec(1, 0),
        subtreeCount(subtreeCount_vec.data(), subtreeCount_vec.size()) {}

void ParallelOctreeTopDownSubtrees::buildOctree(queue &queue, buffer<double> &current_positions_x,
                                                buffer<double> &current_positions_y,
                                                buffer<double> &current_positions_z, buffer<double> &masses,
                                                TimeMeasurement &timer) {
    auto begin = std::chrono::steady_clock::now();

    // compute the axis aligned bounding box around all bodies
    computeMinMaxValuesAABB(queue, current_positions_x, current_positions_y, current_positions_z);
    auto endAABB_creation = std::chrono::steady_clock::now();

    buildOctreeToLevel(queue, current_positions_x, current_positions_y, current_positions_z, masses);
    auto endBuildOctreeToLevel = std::chrono::steady_clock::now();

    d_type::int_t nodeCount; // the node count after the top of the octree has been build
    {
        host_accessor<d_type::int_t> NodeCount(nextFreeNodeID);
        nodeCount = NodeCount[0];
    }
    std::vector<d_type::int_t> bodyCountSubtree_vec(nodeCount, 0);
    buffer<d_type::int_t> bodyCountSubtree(bodyCountSubtree_vec.data(), bodyCountSubtree_vec.size());

    // stores the root nodes of all subtrees
    std::vector<d_type::int_t> subtrees_vec(nodeCount);
    buffer<d_type::int_t> subtrees(subtrees_vec.data(), subtrees_vec.size());

    prepareSubtrees(queue, bodyCountSubtree, subtrees, nodeCount);
    auto endPrepareSubtrees = std::chrono::steady_clock::now();

    host_accessor SUBTREE_COUNT(subtreeCount);
    numberOfSubtrees = SUBTREE_COUNT[0];

    // stores for each subtree where the indices of the bodies belonging to this subtree start
    std::vector<d_type::int_t> bodiesOfSubtreeStartIndex_vec(numberOfSubtrees, 0);
    buffer<d_type::int_t> bodiesOfSubtreeStartIndex(bodiesOfSubtreeStartIndex_vec.data(),
                                                    bodiesOfSubtreeStartIndex_vec.size());

    sortBodiesForSubtrees(queue, bodyCountSubtree, subtrees, bodiesOfSubtreeStartIndex);
    auto endSortBodiesForSubtrees = std::chrono::steady_clock::now();


    buildSubtrees(queue, current_positions_x, current_positions_y,
                  current_positions_z, masses, bodiesOfSubtreeStartIndex, bodyCountSubtree, subtrees);
    auto endBuildSubtrees = std::chrono::steady_clock::now();


    if (queue.get_device().is_gpu()) {
        computeCenterOfMass_GPU(queue, current_positions_x, current_positions_y, current_positions_z, masses);
    } else {
        computeCenterOfMass_CPU(queue, current_positions_x, current_positions_y, current_positions_z, masses);
    }
    auto endCenterOfMass = std::chrono::steady_clock::now();

    if (configuration::barnes_hut_algorithm::sortBodies) {
        sortBodies(queue, current_positions_x, current_positions_y, current_positions_z);
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "---------------------------------------------------------- "
              << std::chrono::duration<double, std::milli>(end - begin).count()
              << std::endl;


    timer.addTimeToSequence("AABB creation",
                            std::chrono::duration<double, std::milli>(endAABB_creation - begin).count());
    timer.addTimeToSequence("Build octree to level", std::chrono::duration<double, std::milli>(
            endBuildOctreeToLevel - endAABB_creation).count());
    timer.addTimeToSequence("Prepare subtrees", std::chrono::duration<double, std::milli>(
            endPrepareSubtrees - endBuildOctreeToLevel).count());
    timer.addTimeToSequence("Sort bodies for subtrees", std::chrono::duration<double, std::milli>(
            endSortBodiesForSubtrees - endPrepareSubtrees).count());
    timer.addTimeToSequence("Build subtrees", std::chrono::duration<double, std::milli>(
            endBuildSubtrees - endSortBodiesForSubtrees).count());
    timer.addTimeToSequence("Compute center of mass",
                            std::chrono::duration<double, std::milli>(endCenterOfMass - endBuildSubtrees).count());

    if (configuration::barnes_hut_algorithm::sortBodies) {
        timer.addTimeToSequence("Sort bodies",
                                std::chrono::duration<double, std::milli>(end - endCenterOfMass).count());
    }
}

void ParallelOctreeTopDownSubtrees::buildOctreeToLevel(queue &queue, buffer<double> &current_positions_x,
                                                       buffer<double> &current_positions_y,
                                                       buffer<double> &current_positions_z, buffer<double> &masses) {
    d_type::int_t N = configuration::numberOfBodies;
    d_type::int_t storageSize = configuration::barnes_hut_algorithm::storageSizeParameter;
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
        accessor<d_type::int_t> OCTANTS(octants, h);
        accessor<double> EDGE_LENGTHS(edgeLengths, h);
        accessor<double> MIN_X(min_x_values, h);
        accessor<double> MIN_Y(min_y_values, h);
        accessor<double> MIN_Z(min_z_values, h);
        accessor<d_type::int_t> NEXT_FREE_NODE_ID(nextFreeNodeID, h);
        accessor<int> NODE_LOCKED(nodeIsLocked, h);
        accessor<int> NODE_IS_LEAF(nodeIsLeaf, h);
        accessor<double> SUM_MASSES(sumOfMasses, h);
        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);
        accessor<d_type::int_t> BODY_OF_NODE(bodyOfNode, h);
        accessor<d_type::int_t> BODY_COUNT_NODE(bodyCountNode, h);

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
            BODY_COUNT_NODE[0] = 0;

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
        accessor<d_type::int_t> OCTANTS(octants, h);
        accessor<double> EDGE_LENGTHS(edgeLengths, h);
        accessor<double> MIN_X(min_x_values, h);
        accessor<double> MIN_Y(min_y_values, h);
        accessor<double> MIN_Z(min_z_values, h);
        accessor<d_type::int_t> BODY_OF_NODE(bodyOfNode, h);
        accessor<double> SUM_MASSES(sumOfMasses, h);
        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);
        accessor<d_type::int_t> NEXT_FREE_NODE_ID(nextFreeNodeID, h);
        accessor<d_type::int_t> SUBTREE_OF_BODY(subtreeOfBody, h);
        accessor<d_type::int_t> BODY_COUNT_NODE(bodyCountNode, h);

        // determine the maximum body count per work-item
        d_type::int_t bodyCountPerWorkItem;
        if (configuration::numberOfBodies > configuration::barnes_hut_algorithm::octreeWorkItemCount) {
            bodyCountPerWorkItem = std::ceil(
                    (double) configuration::numberOfBodies /
                    (double) configuration::barnes_hut_algorithm::octreeWorkItemCount);
        } else {
            bodyCountPerWorkItem = 1;
        }

        auto kernelRange = nd_range<1>(range<1>(configuration::barnes_hut_algorithm::octreeWorkItemCount),
                                       range<1>(configuration::barnes_hut_algorithm::octreeWorkItemCount));

        h.parallel_for(kernelRange, [=](auto &nd_item) {
            atomic_ref<d_type::int_t, memory_order::acq_rel, memory_scope::device,
                    access::address_space::global_space> nextFreeNodeIDAccessor(NEXT_FREE_NODE_ID[0]);

            // for all bodies assigned to this work-item.
            for (d_type::int_t i = nd_item.get_global_id() * bodyCountPerWorkItem;
                 i < (d_type::int_t) (nd_item.get_global_id() * bodyCountPerWorkItem + bodyCountPerWorkItem); ++i) {

                if (i < N) { // check if body ID actually exists
                    SUBTREE_OF_BODY[i] = 0;

                    //start inserting at the root node
                    d_type::int_t currentDepth = 0;
                    d_type::int_t currentNode = 0;

                    bool bodyInserted = false;

                    while (!bodyInserted) {
                        // We check if the current node is locked by some other thread. If it is not, this thread locks it and
                        // continues with the insertion process.
                        int exp = 0; // expected value for compare_exchange
                        atomic_ref<int, memory_order::acq_rel, memory_scope::device, access::address_space::global_space> atomicNodeIsLockedAccessor(
                                NODE_LOCKED[currentNode]);

                        atomic_ref<int, memoryOrderReference, memory_scope::device, access::address_space::global_space> atomicNodeIsLeafAccessor(
                                NODE_IS_LEAF[currentNode]);

                        if (currentDepth < maxLevel) {

                            if (atomicNodeIsLeafAccessor.load(memoryOrderLoad, memory_scope::device) == 1) {
                                // the current node is a leaf node, try to lock the node and insert body

                                if (atomicNodeIsLockedAccessor.compare_exchange_strong(exp, 1, memory_order::acq_rel,
                                                                                       memory_scope::device)) {

                                    if (atomicNodeIsLeafAccessor.load(memoryOrderLoad, memory_scope::device) == 1) {
                                        // node is locked and still a leaf node --> it is safe to continue with insertion

                                        if (BODY_OF_NODE[currentNode] == N) {
                                            // the leaf node is empty --> Insert the current body into the current node and set the flag to continue with next body

                                            BODY_OF_NODE[currentNode] = i;

                                            bodyInserted = true;
                                            nd_item.mem_fence(access::fence_space::global_and_local);
                                        } else {
                                            // the leaf node already contains a body --> split the node and insert old body

                                            d_type::int_t bodyIDinNode = BODY_OF_NODE[currentNode];

                                            // determine insertion index for the new child nodes and reserve 8 indices
                                            d_type::int_t incr = 8;
                                            d_type::int_t firstIndex = nextFreeNodeIDAccessor.fetch_add(incr);

                                            double childEdgeLength = EDGE_LENGTHS[currentNode] / 2;
                                            double parent_min_x = MIN_X[currentNode];
                                            double parent_min_y = MIN_Y[currentNode];
                                            double parent_min_z = MIN_Z[currentNode];

                                            // set the edge lengths of the child nodes
                                            for (d_type::int_t idx = firstIndex;
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
                                            for (d_type::int_t idx = firstIndex; idx < firstIndex + 8; ++idx) {
                                                OCTANTS[5 * storageSize + idx] = 0;
                                                OCTANTS[7 * storageSize + idx] = 0;
                                                OCTANTS[4 * storageSize + idx] = 0;
                                                OCTANTS[6 * storageSize + idx] = 0;
                                                OCTANTS[1 * storageSize + idx] = 0;
                                                OCTANTS[3 * storageSize + idx] = 0;
                                                OCTANTS[0 + idx] = 0;
                                                OCTANTS[2 * storageSize + idx] = 0;

                                                BODY_OF_NODE[idx] = N;
                                                BODY_COUNT_NODE[idx] = 0;
                                                NODE_IS_LEAF[idx] = 1;

                                                CENTER_OF_MASS_X[idx] = 0;
                                                CENTER_OF_MASS_Y[idx] = 0;
                                                CENTER_OF_MASS_Z[idx] = 0;
                                                SUM_MASSES[idx] = 0;
                                            }

                                            // determine the new octant for the old body
                                            d_type::int_t octantID;
                                            double parentMin_x = MIN_X[currentNode];
                                            double parentMin_y = MIN_Y[currentNode];
                                            double parentMin_z = MIN_Z[currentNode];
                                            double parentEdgeLength = EDGE_LENGTHS[currentNode];

                                            bool upperPart =
                                                    POS_Y[bodyIDinNode] > parentMin_y + (parentEdgeLength / 2.0);
                                            bool rightPart =
                                                    POS_X[bodyIDinNode] > parentMin_x + (parentEdgeLength / 2.0);
                                            bool backPart =
                                                    POS_Z[bodyIDinNode] < parentMin_z + (parentEdgeLength / 2.0);

                                            // interpret as binary and convert into decimal
                                            d_type::int_t octantAddress =
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

                                            nd_item.mem_fence(access::fence_space::global_and_local);
                                            // mark the current node as a non leaf node
                                            atomicNodeIsLeafAccessor.store(0, memoryOrderStore,
                                                                           memory_scope::device);
                                        }

                                    }
                                    // release the lock
                                    nd_item.mem_fence(access::fence_space::global_and_local);
                                    atomicNodeIsLockedAccessor.store(0, memoryOrderStore,
                                                                     memory_scope::device);
                                }
                            } else {
                                // the current node is not a leaf node, i.e. it has 8 children
                                // --> determine the octant, the body has to be inserted and set this octant as current node.

                                d_type::int_t octantID;
                                double parentMin_x = MIN_X[currentNode];
                                double parentMin_y = MIN_Y[currentNode];
                                double parentMin_z = MIN_Z[currentNode];
                                double parentEdgeLength = EDGE_LENGTHS[currentNode];

                                bool upperPart = POS_Y[i] > parentMin_y + (parentEdgeLength / 2.0);
                                bool rightPart = POS_X[i] > parentMin_x + (parentEdgeLength / 2.0);
                                bool backPart = POS_Z[i] < parentMin_z + (parentEdgeLength / 2.0);

                                // interpret as binary and convert into decimal
                                d_type::int_t octantAddress = ((int) upperPart) * 4 + ((int) rightPart) * 2 +
                                                              ((int) backPart) * 1;

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
                            bodyInserted = true;

                            // The subtree of the current body i, will be this subtree
                            SUBTREE_OF_BODY[i] = currentNode;
                        }
                    }
                }
            }
        });
    }).wait();

}

void ParallelOctreeTopDownSubtrees::prepareSubtrees(queue &queue, buffer<d_type::int_t> &bodyCountSubtree,
                                                    buffer<d_type::int_t> &subtrees,
                                                    d_type::int_t nodeCount) {
    // iterate over all bodies and determine the body count for each subtree
    queue.submit([&](handler &h) {
        accessor<d_type::int_t> SUBTREE_OF_BODY(subtreeOfBody, h);
        accessor<d_type::int_t> BODY_COUNT_SUBTREE(bodyCountSubtree, h);

        h.parallel_for(sycl::range<1>(configuration::numberOfBodies), [=](auto &i) {

            d_type::int_t currentSubtreeNode = SUBTREE_OF_BODY[i];
            atomic_ref<d_type::int_t, memory_order::relaxed, memory_scope::device,
                    access::address_space::global_space> bodyCounter(BODY_COUNT_SUBTREE[currentSubtreeNode]);
            d_type::int_t incr = 1;
            bodyCounter.fetch_add(incr, memory_order::relaxed, memory_scope::device);

        });
    }).wait();

    // iterate over all nodes and determine the subtrees, i.e. the nodes which are referenced by at least one body as their subtree root node
    queue.submit([&](handler &h) {
        accessor<d_type::int_t> SUBTREES(subtrees, h);
        accessor<d_type::int_t> BODY_COUNT_SUBTREE(bodyCountSubtree, h);
        accessor<d_type::int_t> SUBTREE_COUNT(subtreeCount, h);

        h.single_task([=]() {
            d_type::int_t nextIndex = 0;
            // skip the root node since all bodies that reference the root node as their subtree already have been inserted
            // at a higher level than maxLevel, i.e. are already at their final position and do not have to be inserted into a subtree.
            for (d_type::int_t i = 1; i < nodeCount; ++i) {

                if (BODY_COUNT_SUBTREE[i] >
                    0) { // the current node is referenced by at least one body as the root node of their subtree
                    SUBTREES[nextIndex] = i; // store that the current node is a subtree root node
                    nextIndex += 1;
                }
            }
            SUBTREE_COUNT[0] = nextIndex; // the amount of subtrees that has been found
        });
    }).wait();
}

void ParallelOctreeTopDownSubtrees::sortBodiesForSubtrees(queue &queue, buffer<d_type::int_t> &bodyCountSubtree,
                                                          buffer<d_type::int_t> &subtrees,
                                                          buffer<d_type::int_t> &bodiesOfSubtreeStartIndex) {
    // used to keep track of the next insertion index for the next body for each subtree
    std::vector<d_type::int_t> nextPositions_vec(numberOfSubtrees, 0);
    buffer<d_type::int_t> nextPositions(nextPositions_vec.data(), nextPositions_vec.size());

    queue.submit([&](handler &h) {
        accessor<d_type::int_t> BODY_COUNT_SUBTREE(bodyCountSubtree, h);
        accessor<d_type::int_t> SUBTREES(subtrees, h);
        accessor<d_type::int_t> START_INDEX(bodiesOfSubtreeStartIndex, h);
        accessor<d_type::int_t> NEXT_POSITIONS(nextPositions, h);

        h.parallel_for(sycl::range<1>(numberOfSubtrees), [=](auto &i) {
            // determine first index, i.e. the total amount of bodies in all subtrees that will get stored at indices
            // before the bodies of this subtree. These are the bodies of all subtrees with a subtree index lower than i.
            d_type::int_t firstIndex = 0;
            for (int j = i - 1; j >= 0; j--) {
                firstIndex += BODY_COUNT_SUBTREE[SUBTREES[j]];
            }
            START_INDEX[i] = firstIndex;
            NEXT_POSITIONS[i] = firstIndex;
        });
    }).wait();

    queue.submit([&](handler &h) {
        accessor<d_type::int_t> SUBTREE_OF_BODY(subtreeOfBody, h);
        accessor<d_type::int_t> SORTED_BODIES(sortedBodies, h);
        accessor<d_type::int_t> NEXT_POSITIONS(nextPositions, h);
        accessor<d_type::int_t> SUBTREES(subtrees, h);
        d_type::int_t NUM_SUBTREES = numberOfSubtrees;

        // iterate over all bodies and determine the subtree of each body. After that insert the body at the next free position
        // of the storage space that got allocated for this subtree.
        h.parallel_for(sycl::range<1>(numberOfBodies), [=](auto &i) {
            d_type::int_t subtreeOfCurrentBody = SUBTREE_OF_BODY[i]; // root node of subtree

            // find out index of subtree represented by the subtree root node
            int subtreeIndex = -1;
            for (int j = 0; j < NUM_SUBTREES; ++j) {
                if (SUBTREES[j] == subtreeOfCurrentBody) {
                    subtreeIndex = j;
                    break;
                }
            }

            // insert body at next free position in the part of the sortedBodies buffer reserved for the respective octree
            if (subtreeIndex != -1) {
                atomic_ref<d_type::int_t, memory_order::acq_rel, memory_scope::device,
                        access::address_space::global_space> atomicNextPositionAccessor(NEXT_POSITIONS[subtreeIndex]);
                d_type::int_t incr = 1;
                d_type::int_t insertionIndex = atomicNextPositionAccessor.fetch_add(incr);
                SORTED_BODIES[insertionIndex] = i;
            }
        });
    }).wait();
}


void ParallelOctreeTopDownSubtrees::buildSubtrees(queue &queue, buffer<double> &current_positions_x,
                                                  buffer<double> &current_positions_y,
                                                  buffer<double> &current_positions_z, buffer<double> &masses,
                                                  buffer<d_type::int_t> &bodiesOfSubtreeStartIndex,
                                                  buffer<d_type::int_t> &bodyCountSubtree,
                                                  buffer<d_type::int_t> &subtrees) {


    d_type::int_t N = configuration::numberOfBodies;
    d_type::int_t storageSize = configuration::barnes_hut_algorithm::storageSizeParameter;

    d_type::int_t maxLevel = configuration::barnes_hut_algorithm::maxBuildLevel;
    d_type::int_t localSize = configuration::barnes_hut_algorithm::octreeWorkItemCount;

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

    // build subtrees in parallel
    queue.submit([&](handler &h) {
        accessor<int> NODE_LOCKED(nodeIsLocked, h);
        accessor<int> NODE_IS_LEAF(nodeIsLeaf, h);
        accessor<double> POS_X(current_positions_x, h);
        accessor<double> POS_Y(current_positions_y, h);
        accessor<double> POS_Z(current_positions_z, h);
        accessor<d_type::int_t> OCTANTS(octants, h);
        accessor<double> EDGE_LENGTHS(edgeLengths, h);
        accessor<double> MIN_X(min_x_values, h);
        accessor<double> MIN_Y(min_y_values, h);
        accessor<double> MIN_Z(min_z_values, h);
        accessor<d_type::int_t> BODY_OF_NODE(bodyOfNode, h);
        accessor<double> SUM_MASSES(sumOfMasses, h);
        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);
        accessor<d_type::int_t> NEXT_FREE_NODE_ID(nextFreeNodeID, h);
        accessor<d_type::int_t> START_INDEX(bodiesOfSubtreeStartIndex, h);
        accessor<d_type::int_t> BODY_COUNT_SUBTREE(bodyCountSubtree, h);
        accessor<d_type::int_t> SORTED_BODIES(sortedBodies, h);
        accessor<d_type::int_t> SUBTREES(subtrees, h);
        accessor<d_type::int_t> BODY_COUNT_NODE(bodyCountNode, h);

        auto kernelRange = nd_range<1>(range<1>(numberOfSubtrees * localSize), range<1>(localSize));

        h.parallel_for(kernelRange, [=](auto &nd_item) {

            atomic_ref<d_type::int_t, memory_order::acq_rel, memory_scope::device, access::address_space::global_space> nextFreeNodeIDAccessor(
                    NEXT_FREE_NODE_ID[0]);

            d_type::int_t workGroupID = nd_item.get_group_linear_id();
            d_type::int_t subTreeRootNode = SUBTREES[workGroupID];
            d_type::int_t bodyCountWorkGroup = BODY_COUNT_SUBTREE[subTreeRootNode];
            d_type::int_t startIndexBodies = START_INDEX[workGroupID];

            // determine the body count for each work item
            d_type::int_t bodyCountPerWorkItem;
            if (bodyCountWorkGroup > localSize) {
                bodyCountPerWorkItem = sycl::ceil((double) bodyCountWorkGroup / (double) localSize);
            } else {
                bodyCountPerWorkItem = 1;
                if (nd_item.get_local_id() > bodyCountWorkGroup) {
                    return;
                }
            }

            // for all bodies assigned to this work-item.
            for (d_type::int_t index = startIndexBodies + nd_item.get_local_id() * bodyCountPerWorkItem;
                 index < (d_type::int_t) (startIndexBodies + nd_item.get_local_id() * bodyCountPerWorkItem +
                                          bodyCountPerWorkItem); ++index) {

                // check if body ID is actually part of the body IDs of this work group
                if (index < (startIndexBodies + bodyCountWorkGroup)) {
                    // the id of the current body
                    d_type::int_t i = SORTED_BODIES[index];

                    //start inserting at the subtree root node
                    d_type::int_t currentDepth = maxLevel;
                    d_type::int_t currentNode = subTreeRootNode;

                    bool bodyInserted = false;

                    while (!bodyInserted) {
                        int exp = 0; // expected value for compare_exchange
                        atomic_ref<int, memory_order::acq_rel, memory_scope::device, access::address_space::global_space> atomicNodeIsLockedAccessor(
                                NODE_LOCKED[currentNode]);

                        atomic_ref<int, memoryOrderReference, memory_scope::device, access::address_space::global_space> atomicNodeIsLeafAccessor(
                                NODE_IS_LEAF[currentNode]);

                        if (atomicNodeIsLeafAccessor.load(memoryOrderLoad, memory_scope::device) == 1) {
                            // the current node is a leaf node, try to lock the node and insert body

                            // It is checked if the current node is locked by some other thread. If it is not, this thread locks it and
                            // continues with the insertion process.
                            if (atomicNodeIsLockedAccessor.compare_exchange_strong(exp, 1, memory_order::acq_rel,
                                                                                   memory_scope::device)) {

                                if (atomicNodeIsLeafAccessor.load(memoryOrderLoad, memory_scope::device) == 1) {
                                    // node is locked and still a leaf node --> it is safe to continue with insertion

                                    if (BODY_OF_NODE[currentNode] == N) {
                                        // the leaf node is empty --> Insert the current body into the current node and set the flag to continue with next body
                                        BODY_OF_NODE[currentNode] = i;

                                        bodyInserted = true;
                                        nd_item.mem_fence(access::fence_space::global_and_local);
                                    } else {
                                        // the leaf node already contains a body --> split the node and insert old body

                                        d_type::int_t bodyIDinNode = BODY_OF_NODE[currentNode];

                                        // determine insertion index for the new child nodes and reserve 8 indices
                                        d_type::int_t incr = 8;
                                        d_type::int_t firstIndex = nextFreeNodeIDAccessor.fetch_add(incr);

                                        double childEdgeLength = EDGE_LENGTHS[currentNode] / 2;
                                        double parent_min_x = MIN_X[currentNode];
                                        double parent_min_y = MIN_Y[currentNode];
                                        double parent_min_z = MIN_Z[currentNode];

                                        // set the edge lengths of the child nodes
                                        for (d_type::int_t idx = firstIndex;
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
                                        for (d_type::int_t idx = firstIndex;
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
                                            BODY_COUNT_NODE[idx] = 0;
                                            NODE_IS_LEAF[idx] = 1;
                                            CENTER_OF_MASS_X[idx] = 0;
                                            CENTER_OF_MASS_Y[idx] = 0;
                                            CENTER_OF_MASS_Z[idx] = 0;
                                            SUM_MASSES[idx] = 0;
                                        }

                                        // determine the new octant for the old body
                                        d_type::int_t octantID;
                                        double parentMin_x = MIN_X[currentNode];
                                        double parentMin_y = MIN_Y[currentNode];
                                        double parentMin_z = MIN_Z[currentNode];
                                        double parentEdgeLength = EDGE_LENGTHS[currentNode];

                                        bool upperPart = POS_Y[bodyIDinNode] > parentMin_y + (parentEdgeLength / 2);
                                        bool rightPart = POS_X[bodyIDinNode] > parentMin_x + (parentEdgeLength / 2);
                                        bool backPart = POS_Z[bodyIDinNode] < parentMin_z + (parentEdgeLength / 2);

                                        // interpret as binary and convert into decimal
                                        d_type::int_t octantAddress = ((int) upperPart) * 4 + ((int) rightPart) * 2 +
                                                                      ((int) backPart) * 1;

                                        // find start index of octant type
                                        octantAddress = octantAddress * storageSize;

                                        // get octant of current Node
                                        octantAddress = octantAddress + currentNode;

                                        octantID = OCTANTS[octantAddress];

                                        // insert the old body into the new octant it belongs to and remove it from the parent node
                                        BODY_OF_NODE[octantID] = bodyIDinNode;
                                        BODY_OF_NODE[currentNode] = N;

                                        nd_item.mem_fence(access::fence_space::global_and_local);
                                        // mark the current node as a non leaf node
                                        atomicNodeIsLeafAccessor.store(0, memoryOrderStore, memory_scope::device);
                                    }
                                }
                                // release the lock
                                nd_item.mem_fence(access::fence_space::global_and_local);
                                atomicNodeIsLockedAccessor.store(0, memoryOrderStore, memory_scope::device);
                            }
                        } else {
                            // the current node is not a leaf node, i.e. it has 8 children
                            // --> determine the octant, the body has to be inserted and set this octant as current node.

                            d_type::int_t octantID;
                            double parentMin_x = MIN_X[currentNode];
                            double parentMin_y = MIN_Y[currentNode];
                            double parentMin_z = MIN_Z[currentNode];
                            double parentEdgeLength = EDGE_LENGTHS[currentNode];

                            bool upperPart = POS_Y[i] > parentMin_y + (parentEdgeLength / 2);
                            bool rightPart = POS_X[i] > parentMin_x + (parentEdgeLength / 2);
                            bool backPart = POS_Z[i] < parentMin_z + (parentEdgeLength / 2);

                            // interpret as binary and convert into decimal
                            d_type::int_t octantAddress =
                                    ((int) upperPart) * 4 + ((int) rightPart) * 2 + ((int) backPart) * 1;

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
                }
            }
        });
    }).wait();
}