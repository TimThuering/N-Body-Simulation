#include "ParallelOctreeTopDownSynchronized.hpp"
#include "Configuration.hpp"
#include "Definitions.hpp"

ParallelOctreeTopDownSynchronized::ParallelOctreeTopDownSynchronized() :
        BarnesHutOctree(),
        nodeIsLocked_vec(configuration::barnes_hut_algorithm::storageSizeParameter, 0),
        nodeIsLocked(nodeIsLocked_vec.data(), nodeIsLocked_vec.size()) {}


void ParallelOctreeTopDownSynchronized::buildOctree(queue &queue, buffer<double> &current_positions_x,
                                                    buffer<double> &current_positions_y,
                                                    buffer<double> &current_positions_z, buffer<double> &masses, TimeMeasurement &timer) {

    auto begin = std::chrono::steady_clock::now();

    // compute the axis aligned bounding box around all bodies
    computeMinMaxValuesAABB(queue, current_positions_x, current_positions_y, current_positions_z);

    auto endAABB_creation = std::chrono::steady_clock::now();

    std::vector<std::size_t> maxTreeDepth_vec(1, 0);
    buffer<std::size_t> maxTreeDepths(maxTreeDepth_vec.data(), maxTreeDepth_vec.size());

    std::size_t N = configuration::numberOfBodies;
    std::size_t storageSize = configuration::barnes_hut_algorithm::storageSizeParameter;


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
        accessor<std::size_t> BODY_COUNT_NODE(bodyCountNode, h);


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
        accessor<double> MASSES(masses, h);
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
        accessor<std::size_t> BODY_COUNT_NODE(bodyCountNode, h);


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

                                if (atomicNodeIsLeafAccessor.load(memoryOrderLoad, memory_scope::device) == 1) {
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

                                                // update sum masses and center of mass
//                                                atomic_ref<double, memory_order::relaxed, memory_scope::device,
//                                                        access::address_space::global_space> atomicMasses(
//                                                        SUM_MASSES[currentNode]);
//
//                                                atomic_ref<double, memory_order::relaxed, memory_scope::device,
//                                                        access::address_space::global_space> atomicCenterMass_X(
//                                                        CENTER_OF_MASS_X[currentNode]);
//
//                                                atomic_ref<double, memory_order::relaxed, memory_scope::device,
//                                                        access::address_space::global_space> atomicCenterMass_Y(
//                                                        CENTER_OF_MASS_Y[currentNode]);
//
//                                                atomic_ref<double, memory_order::relaxed, memory_scope::device,
//                                                        access::address_space::global_space> atomicCenterMass_Z(
//                                                        CENTER_OF_MASS_Z[currentNode]);
//
//                                                atomicMasses.fetch_add(MASSES[i], memory_order::relaxed,
//                                                                       memory_scope::device);
//                                                atomicCenterMass_X.fetch_add(POS_X[i] * MASSES[i],
//                                                                             memory_order::relaxed,
//                                                                             memory_scope::device);
//                                                atomicCenterMass_Y.fetch_add(POS_Y[i] * MASSES[i],
//                                                                             memory_order::relaxed,
//                                                                             memory_scope::device);
//                                                atomicCenterMass_Z.fetch_add(POS_Z[i] * MASSES[i],
//                                                                             memory_order::relaxed,
//                                                                             memory_scope::device);


                                                nodeInserted = true;
                                                //sycl::atomic_fence(memory_order::acq_rel, memory_scope::device);
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

                                                    BODY_COUNT_NODE[idx] = 0;

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
                                                        POS_Y[bodyIDinNode] > parentMin_y + (parentEdgeLength / 2);
                                                bool rightPart =
                                                        POS_X[bodyIDinNode] > parentMin_x + (parentEdgeLength / 2);
                                                bool backPart =
                                                        POS_Z[bodyIDinNode] < parentMin_z + (parentEdgeLength / 2);

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


//                                                atomic_ref<double, memory_order::relaxed, memory_scope::device,
//                                                        access::address_space::global_space> atomicMasses(
//                                                        SUM_MASSES[octantID]);
//
//                                                atomic_ref<double, memory_order::relaxed, memory_scope::device,
//                                                        access::address_space::global_space> atomicCenterMass_X(
//                                                        CENTER_OF_MASS_X[octantID]);
//
//                                                atomic_ref<double, memory_order::relaxed, memory_scope::device,
//                                                        access::address_space::global_space> atomicCenterMass_Y(
//                                                        CENTER_OF_MASS_Y[octantID]);
//
//                                                atomic_ref<double, memory_order::relaxed, memory_scope::device,
//                                                        access::address_space::global_space> atomicCenterMass_Z(
//                                                        CENTER_OF_MASS_Z[octantID]);
//
//
//                                                atomicMasses.fetch_add(MASSES[bodyIDinNode], memory_order::relaxed,
//                                                                       memory_scope::device);
//                                                atomicCenterMass_X.fetch_add(POS_X[bodyIDinNode] * MASSES[bodyIDinNode],
//                                                                             memory_order::relaxed,
//                                                                             memory_scope::device);
//                                                atomicCenterMass_Y.fetch_add(POS_Y[bodyIDinNode] * MASSES[bodyIDinNode],
//                                                                             memory_order::relaxed,
//                                                                             memory_scope::device);
//                                                atomicCenterMass_Z.fetch_add(POS_Z[bodyIDinNode] * MASSES[bodyIDinNode],
//                                                                             memory_order::relaxed,
//                                                                             memory_scope::device);


                                                //sycl::atomic_fence(memory_order::acq_rel, memory_scope::device);
                                                nd_item.mem_fence(access::fence_space::global_and_local);
                                                // mark the current node as a non leaf node
                                                atomicNodeIsLeafAccessor.store(0, memoryOrderStore,
                                                                               memory_scope::device);
                                            }
                                            // sycl::atomic_fence(memory_order::acq_rel, memory_scope::device);
                                            //nd_item.mem_fence(access::fence_space::global_and_local);
                                        }
                                        // release the lock
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
                                            ((int) upperPart) * 4 + ((int) rightPart) * 2 + ((int) backPart) * 1;

                                    // find start index of octant type
                                    octantAddress = octantAddress * storageSize;

                                    // get octant of current Node
                                    octantAddress = octantAddress + currentNode;

                                    octantID = OCTANTS[octantAddress];

                                    // update sum masses and center of mass of this node, since the current body will be inserted in one of the children
//                                    atomic_ref<double, memory_order::relaxed, memory_scope::device,
//                                            access::address_space::global_space> atomicMasses(
//                                            SUM_MASSES[currentNode]);
//
//                                    atomic_ref<double, memory_order::relaxed, memory_scope::device,
//                                            access::address_space::global_space> atomicCenterMass_X(
//                                            CENTER_OF_MASS_X[currentNode]);
//
//                                    atomic_ref<double, memory_order::relaxed, memory_scope::device,
//                                            access::address_space::global_space> atomicCenterMass_Y(
//                                            CENTER_OF_MASS_Y[currentNode]);
//
//                                    atomic_ref<double, memory_order::relaxed, memory_scope::device,
//                                            access::address_space::global_space> atomicCenterMass_Z(
//                                            CENTER_OF_MASS_Z[currentNode]);
//
//
//                                    atomicMasses.fetch_add(MASSES[i], memory_order::relaxed, memory_scope::device);
//                                    atomicCenterMass_X.fetch_add(POS_X[i] * MASSES[i], memory_order::relaxed,
//                                                                 memory_scope::device);
//                                    atomicCenterMass_Y.fetch_add(POS_Y[i] * MASSES[i], memory_order::relaxed,
//                                                                 memory_scope::device);
//                                    atomicCenterMass_Z.fetch_add(POS_Z[i] * MASSES[i], memory_order::relaxed,
//                                                                 memory_scope::device);

                                    currentNode = octantID;
                                    currentDepth += 1;
                                }
                                //nd_item.barrier();
                            }
                        }
                    }
                });
    }).wait();

    auto endBuildOctree = std::chrono::steady_clock::now();


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
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << std::endl;

    timer.addTimeToSequence("AABB creation", std::chrono::duration<double, std::milli>(endAABB_creation - begin).count());
    timer.addTimeToSequence("Build octree", std::chrono::duration<double, std::milli>(endBuildOctree - endAABB_creation).count());
    timer.addTimeToSequence("Compute center of mass", std::chrono::duration<double, std::milli>(endCenterOfMass - endBuildOctree).count());

    if (configuration::barnes_hut_algorithm::sortBodies) {
        timer.addTimeToSequence("Sort bodies", std::chrono::duration<double, std::milli>(end - endCenterOfMass).count());
    }

//    host_accessor CX(massCenters_x);
//    host_accessor CY(massCenters_y);
//    host_accessor CZ(massCenters_z
//    );
//    std::cout << MASSES[0] << std::endl;
//    std::cout << CX[0] << std::endl;
//    std::cout << CY[0] << std::endl;
//    std::cout << CZ[0] << std::endl;

}

void ParallelOctreeTopDownSynchronized::computeCenterOfGravity(queue &queue, buffer<double> &current_positions_x,
                                                               buffer<double> &current_positions_y,
                                                               buffer<double> &current_positions_z,
                                                               buffer<double> &masses) {
    host_accessor<std::size_t> NUM_NODES(nextFreeNodeID);
    std::size_t numberOfNodes = NUM_NODES[0];

    std::size_t N = configuration::numberOfBodies;
    std::size_t storageSize = configuration::barnes_hut_algorithm::storageSizeParameter;


    std::size_t nodeCountPerWorkItem;
    if (numberOfNodes > configuration::barnes_hut_algorithm::octreeWorkItemCount) {
        nodeCountPerWorkItem = std::ceil(
                (double) numberOfNodes /
                (double) configuration::barnes_hut_algorithm::octreeWorkItemCount);
    } else {
        nodeCountPerWorkItem = 1;
    }

    std::size_t workItemCount = configuration::barnes_hut_algorithm::octreeWorkItemCount;


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

