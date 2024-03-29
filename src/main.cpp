#include "SimulationData.hpp"
#include "InputParser.hpp"
#include "NaiveAlgorithm.hpp"
#include "BarnesHutAlgorithm.hpp"
#include "TimeConverter.hpp"
#include "Configuration.hpp"
#include <cxxopts.hpp>

int main(int argc, char *argv[]) {
    cxxopts::Options arguments("N-Body-Simulation");

    // add the program arguments
    arguments.add_options()
            ("file", "Path to a .csv file containing the data for the simulation",
             cxxopts::value<std::string>());

    arguments.add_options()
            ("dt", "Width of the time step for the simulation",
             cxxopts::value<std::string>());

    arguments.add_options()
            ("t_end", "t_end specifies the time until the system will be simulated",
             cxxopts::value<std::string>());

    arguments.add_options()
            ("vs", "The time step width of the visualization",
             cxxopts::value<std::string>());

    arguments.add_options()
            ("vs_dir", "The top-level output directory for the ParaView output files",
             cxxopts::value<std::string>());

    arguments.add_options()
            ("theta", "The theta-value which determines the accuracy of the Barnes-Hut algorithm",
             cxxopts::value<double>());

    arguments.add_options()
            ("num_wi_octree", "Determines the number of work-items used to build the octree",
             cxxopts::value<int>());

    arguments.add_options()
            ("num_wi_top_octree", "Determines the number of work-items used to build the top of the octree",
             cxxopts::value<int>());

    arguments.add_options()
            ("num_wi_AABB", "Determines the number of work-items used to calculate the AABB",
             cxxopts::value<int>());

    arguments.add_options()
            ("num_wi_com", "Determines the number of work-items used to calculate the center of mass",
             cxxopts::value<int>());

    arguments.add_options()
            ("max_level_top_octree", "Determines the maximum level to which the top of the octree gets build",
             cxxopts::value<int>());

    arguments.add_options()
            ("storage_size_param", "Scales the amount of memory for the octree data structures",
             cxxopts::value<int>());

    arguments.add_options()
            ("stack_size_param", "Scales the amount of memory for the stack used to traverse the octree",
             cxxopts::value<int>());

    arguments.add_options()
            ("block_size",
             "Determines the size of the blocks after which the local memory is updated in the naive algorithm",
             cxxopts::value<int>());

    arguments.add_options()
            ("algorithm", "The algorithm to use for the simulation either <naive> or <BarnesHut>",
             cxxopts::value<std::string>());

    arguments.add_options()
            ("energy",
             "Turn on energy computation of the system for each visualized time step. Can increase the runtime for large amounts of bodies.",
             cxxopts::value<bool>());

    arguments.add_options()
            ("sort_bodies",
             "Turns on sorting of the bodies according to their in-order position in the octree",
             cxxopts::value<bool>());

    arguments.add_options()
            ("use_gpus",
             "If true, GPUs will be used for the computation. If false CPUs will be used.",
             cxxopts::value<bool>());

    arguments.add_options()
            ("wg_size_barnes_hut",
             "Determines the work-group size of the acceleration kernel in the Barnes-Hut algorithm",
             cxxopts::value<int>());

    arguments.add_options()
            ("opt_stage",
             "Can be either 0,1 or 2. A value of two will correspond to using the highest optimized version of the acceleration kernel of the naive Algorithm. A value of 0 will correspond to using a non-optimized version of this kernel",
             cxxopts::value<int>());

    auto options = arguments.parse(argc, argv);

    // input and output paths
    std::string path = options["file"].as<std::string>();
    std::string outputDirectoryPath = options["vs_dir"].as<std::string>();

    // time values for the simulation
    std::string dt_input = options["dt"].as<std::string>();
    std::string t_end_input = options["t_end"].as<std::string>();
    std::string vs_input = options["vs"].as<std::string>();

    // convert input to double values in earth days
    double dt = TimeConverter::convertToEarthDays(dt_input);
    double t_end = TimeConverter::convertToEarthDays(t_end_input);
    double visualizationStepWidth = TimeConverter::convertToEarthDays(vs_input);

    // Storage for simulation the data
    SimulationData simulationData;

    // parse the csv file containing the simulation data
    InputParser::parse_input(path, simulationData);


    int storageSizeParam;
    if (options.count("storage_size_param") == 1) {
        storageSizeParam = options["storage_size_param"].as<int>();
    } else {
        storageSizeParam = 16;
    }

    int stackSizeParam;
    if (options.count("stack_size_param") == 1) {
        stackSizeParam = options["stack_size_param"].as<int>();
    } else {
        stackSizeParam = 16;
    }

    configuration::initializeConfigValues(simulationData.mass.size(), storageSizeParam, stackSizeParam);

    if (options.count("energy") == 1) {
        configuration::setEnergyComputation(options["energy"].as<bool>());
    }

    if (options.count("use_gpus") == 1) {
        configuration::setDeviceGPU(options["use_gpus"].as<bool>());
    }


    if (options["algorithm"].as<std::string>() == "naive") {
        // overwrite default values of naive algorithm if specified as program argument

        if (options.count("block_size") == 1) {
            configuration::setBlockSize(options["block_size"].as<int>());
        }

        if (options.count("opt_stage") == 1) {
            if (options["opt_stage"].as<int>() > 2 || options["opt_stage"].as<int>() < 0) {
                throw std::invalid_argument("Optimization stage must be 0,1 or 2");
            }
            configuration::setOptimizationStage(options["opt_stage"].as<int>());
        }

        std::cout << "Naive algorithm configuration:" << std::endl;
        std::cout << "Block Size ------------------------------------ " << configuration::naive_algorithm::blockSize
                  << std::endl;
        std::cout << "Optimization stage acceleration kernel -------- " << configuration::naive_algorithm::optimization_stage
                  << std::endl;
        std::cout << std::endl << std::endl;


    } else if (options["algorithm"].as<std::string>() == "BarnesHut") {
        // overwrite default values of Barnes-Hut algorithm if specified as program argument

        if (options.count("theta") == 1) {
            configuration::setTheta(options["theta"].as<double>());
        }

        if (options.count("num_wi_octree") == 1) {
            configuration::setOctreeWorkItemCount(options["num_wi_octree"].as<int>());
        }

        if (options.count("num_wi_top_octree") == 1) {
            configuration::setOctreeTopWorkItemCount(options["num_wi_top_octree"].as<int>());
        }

        if (options.count("num_wi_com") == 1) {
            configuration::setCenterOfMassWorkItemCount(options["num_wi_com"].as<int>());
        }

        if (options.count("max_level_top_octree") == 1) {
            configuration::setMaxBuildLevel(options["max_level_top_octree"].as<int>());
        }

        if (options.count("num_wi_AABB") == 1) {
            configuration::setAABBWorkItemCount(options["num_wi_AABB"].as<int>());
        }

        if (options.count("sort_bodies")) {
            configuration::setSortBodies(options["sort_bodies"].as<bool>());
        }

        if (options.count("wg_size_barnes_hut")) {
            configuration::setWorkGroupSizeBarnesHut(options["wg_size_barnes_hut"].as<int>());
        }


        std::cout << "Barnes-Hut algorithm configuration:" << std::endl;
        std::cout << "Theta ----------------------------------- " << configuration::barnes_hut_algorithm::theta
                  << std::endl;
        std::cout << "Work-items AABB creation ---------------- "
                  << configuration::barnes_hut_algorithm::AABBWorkItemCount << std::endl;
        std::cout << "Work-items octree creation -------------- "
                  << configuration::barnes_hut_algorithm::octreeWorkItemCount << std::endl;
        std::cout << "Work-items center of mass calculation --- "
                  << configuration::barnes_hut_algorithm::centerOfMassWorkItemCount << std::endl;
#ifndef OCTREE_TOP_DOWN_SYNC
        std::cout << "Work-items top of octree creation ------- "
                  << configuration::barnes_hut_algorithm::octreeTopWorkItemCount << std::endl;
        std::cout << "Maximum build level top of octree ------- " << configuration::barnes_hut_algorithm::maxBuildLevel
                  << std::endl;
#endif
        std::cout << "Work-group size acceleration kernel ----- " << configuration::barnes_hut_algorithm::workGroupSize
                  << std::endl;
        std::cout << "Sort Bodies enabled --------------------- " << configuration::barnes_hut_algorithm::sortBodies
                  << std::endl;
        std::cout << std::endl << std::endl;

    } else {
        throw std::invalid_argument("Algorithm must either be <naive> or <BarnesHut>");
    }

    if (options["algorithm"].as<std::string>() == "naive") {
        NaiveAlgorithm algorithm(dt, t_end, visualizationStepWidth, outputDirectoryPath);
        algorithm.startSimulation(simulationData);
        algorithm.generateParaViewOutput(simulationData);
    } else {
        BarnesHutAlgorithm algorithm(dt, t_end, visualizationStepWidth, outputDirectoryPath);
        algorithm.startSimulation(simulationData);
        algorithm.generateParaViewOutput(simulationData);
    }

}
