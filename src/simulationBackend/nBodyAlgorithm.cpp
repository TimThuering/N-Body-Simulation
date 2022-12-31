#include "nBodyAlgorithm.hpp"
#include <iostream>
#include <ctime>
#include <algorithm>
#include <filesystem>
#include "TimeMeasurement.hpp"

void nBodyAlgorithm::computeEnergy(queue &queue, buffer<double> &masses, std::size_t currentStep,
                                   buffer<double> &currentPositions_x,
                                   buffer<double> &currentPositions_y, buffer<double>
                                   &currentPositions_z,
                                   buffer<double> &currentVelocities_x, buffer<double>
                                   &currentVelocities_y,
                                   buffer<double> &currentVelocities_z) {
    double E_kin_result = 0;
    double E_pot_result = 0;
    double E_total_result;


    std::vector<double> E_kin(numberOfBodies);
    std::vector<double> E_pot(numberOfBodies);

    buffer<double> E_kin_values = E_kin;
    buffer<double> E_pot_values = E_pot;

    double G = this->G;

    queue.submit([&](handler &h) {

        accessor<double> V_X(currentVelocities_x, h);
        accessor<double> V_Y(currentVelocities_y, h);
        accessor<double> V_Z(currentVelocities_z, h);

        accessor<double> P_X(currentPositions_x, h);
        accessor<double> P_Y(currentPositions_y, h);
        accessor<double> P_Z(currentPositions_z, h);

        accessor<double> E_KIN(E_kin_values, h);
        accessor<double> E_POT(E_pot_values, h);

        accessor<double> M(masses, h);


        // parallel computation for kinetic and potential energy.
        h.parallel_for(numberOfBodies, [=](auto &j) {
            double v = V_X[j] * V_X[j] +
                       V_Y[j] * V_Y[j] +
                       V_Z[j] * V_Z[j];
            E_KIN[j] = 0.5 * M[j] * v;
            E_POT[j] = 0;

            for (int i = 0; i < j; ++i) {
                double r_x = P_X[j] - P_X[i];
                double r_y = P_Y[j] - P_Y[i];
                double r_z = P_Z[j] - P_Z[i];

                double r = sycl::sqrt(r_x * r_x + r_y * r_y + r_z * r_z);

                E_POT[j] += G * M[i] * M[j] / r;
            }
        });
    }).wait();


    host_accessor<double> E_KIN(E_kin_values);
    host_accessor<double> E_POT(E_pot_values);

    for (std::size_t i = 0; i < numberOfBodies; ++i) {
        E_kin_result += E_KIN[i];
        E_pot_result += E_POT[i];
    }


    E_pot_result *= -1;
    // total energy
    E_total_result = E_kin_result + E_pot_result;

    // store results for ParaView output generation.
    totalEnergy[currentStep] = E_total_result;
    potentialEnergy[currentStep] = E_pot_result;
    kineticEnergy[currentStep] = E_kin_result;
    virialEquilibrium[currentStep] = (2.0 * E_kin_result) / std::abs(E_pot_result);
}

void nBodyAlgorithm::storeAccelerations(std::size_t currentStep, buffer<double> &acceleration_x,
                                        buffer<double> &acceleration_y, buffer<double> &acceleration_z) {

    host_accessor<double> ACC_X(acceleration_x);
    host_accessor<double> ACC_Y(acceleration_y);
    host_accessor<double> ACC_Z(acceleration_z);

    for (std::size_t i = 0; i < numberOfBodies; ++i) {
        double accelerationNorm = ACC_X[i] * ACC_X[i] +
                                  ACC_Y[i] * ACC_Y[i] +
                                  ACC_Z[i] * ACC_Z[i];
        acceleration[currentStep].push_back(std::sqrt(accelerationNorm));
    }

}

void nBodyAlgorithm::adjustVelocities(const SimulationData &simulationData) {
    double sumMasses = 0;
    double sumMassesVelocity_x = 0;
    double sumMassesVelocity_y = 0;
    double sumMassesVelocity_z = 0;

    for (int i = 0; i < numberOfBodies; ++i) {
        sumMasses += simulationData.mass[i];
        sumMassesVelocity_x += simulationData.mass[i] * simulationData.velocities_x[i];
        sumMassesVelocity_y += simulationData.mass[i] * simulationData.velocities_y[i];
        sumMassesVelocity_z += simulationData.mass[i] * simulationData.velocities_z[i];
    }

    double ui_x = sumMassesVelocity_x / sumMasses;
    double ui_y = sumMassesVelocity_y / sumMasses;
    double ui_z = sumMassesVelocity_z / sumMasses;


    for (int i = 0; i < numberOfBodies; ++i) {
        velocities_x[0][i] -= ui_x;
        velocities_y[0][i] -= ui_y;
        velocities_z[0][i] -= ui_z;
    }
}

void nBodyAlgorithm::generateParaViewOutput(const SimulationData &simulationData) {

    // create timestamp for directory name of output files
    std::time_t currentTime = std::time(nullptr);
    std::string time = static_cast<std::string>(std::ctime(&currentTime));
    std::replace(time.begin(), time.end(), ' ', '_');
    time = time.substr(0, time.size() - 1);

    std::string filePathBase = outputDirectory + '/' + time + '/';

    // create the directory for the output files
    std::filesystem::create_directory(filePathBase);

    // export the all timings as json file
    timer.exportJSON(filePathBase + "times.json");

    // generate the .pvd file for this simulation
    std::ofstream pvdFile(filePathBase + "/simulation" + ".pvd");

    // write the header to the .pvd file
    pvdFile << "<?xml version=\"1.0\"?>" << '\n'
            << R"(<VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">)"
            << '\n' << "<Collection>"  << '\n';


    for (std::size_t i = 0; i < positions_x.size(); ++i) {

        // generate file for simulation step i: simulation_step<i>.vtp
        std::string vtpFileName = "simulation_step" + std::to_string(i) + ".vtp";
        std::ofstream vtpFile(filePathBase + vtpFileName);

        // add a reference of the .vtp file to the .pvd file
        pvdFile << "<DataSet timestep=\"" << i << "\" "
//                << R"(group="" part="0" file=")" << time.append("/" + vtpFileName) << "\"/>";
                << R"(group="" part="0" file=")" << vtpFileName << "\"/>" << '\n';

        // write to the file
        writeFileHeader(i, vtpFile);

        writePositions(i, vtpFile);

        vtpFile << "</DataArray>" << '\n'
                << "</Points>" << '\n'
                << "<PointData>" << '\n'
                << R"(<DataArray type="Int32" Name="body_id" NumberOfComponents="1" format="ascii">)" << '\n';

        writeIDs(i, vtpFile);

        vtpFile << "</DataArray>" << '\n'
                << R"(<DataArray type="Float64" Name="velocity" NumberOfComponents="3" format="ascii">)" << '\n';

        writeVelocities(i, vtpFile);

        vtpFile << "</DataArray>" << '\n'
                << R"(<DataArray type="Float64" Name="acceleration" NumberOfComponents="1" format="ascii">)" << '\n';

        writeAccelerations(i, vtpFile);

        vtpFile << "</DataArray>" << '\n'
                << R"(<DataArray type="Float64" Name="mass" NumberOfComponents="1" format="ascii">)" << '\n';

        writeMasses(simulationData, vtpFile);

        vtpFile << "</DataArray>" << '\n'
                << R"(<DataArray type="String" Name="name" NumberOfComponents="1" format="ascii">)" << '\n';

        writeNamesASCII(simulationData, vtpFile);

        vtpFile << "</DataArray>" << '\n'
                << R"(<DataArray type="Int32" Name="orbit_class" NumberOfComponents="1" format="ascii">)" << '\n';

        writeBodyClasses(simulationData, vtpFile);

        vtpFile << "</DataArray>" << '\n'
                << "</PointData>" << '\n'
                << "<Verts>" << '\n'
                << R"(<DataArray type="Int64" Name="offsets">)" << '\n';

        writeOffsets(i, vtpFile);

        vtpFile << '\n' << "</DataArray>" << '\n'
                << R"(<DataArray type="Int64" Name="connectivity">)" << '\n';

        writeConnectivity(i, vtpFile);

        vtpFile << '\n' << "</DataArray>" << '\n'
                << "</Verts>" << '\n'
                << "</Piece>" << '\n'
                << "<FieldData>" << '\n'
                << R"(<DataArray type="Float64" Name="kinetic energy" NumberOfTuples="1" format="ascii">)" << '\n';

        writeKineticEnergy(i, vtpFile);

        vtpFile << "</DataArray>" << '\n'
                << R"(<DataArray type="Float64" Name="potential energy" NumberOfTuples="1" format="ascii">)" << '\n';

        writePotentialEnergy(i, vtpFile);

        vtpFile << "</DataArray>" << '\n'
                << R"(<DataArray type="Float64" Name="total energy" NumberOfTuples="1" format="ascii">)" << '\n';

        writeTotalEnergy(i, vtpFile);

        vtpFile << "</DataArray>" << '\n'
                << R"(<DataArray type="Float64" Name="virial equilibrium" NumberOfTuples="1" format="ascii">)" << '\n';

        writeVirialEquilibrium(i, vtpFile);

        vtpFile << "</DataArray>" << '\n'
                << "</FieldData>" << '\n'
                << "</PolyData>" << '\n'
                << "</VTKFile>" << '\n';

        vtpFile.close();
    }

    // close the .pvd file
    pvdFile << "</Collection>" << '\n' << "</VTKFile>" << '\n';
    pvdFile.close();

}

void nBodyAlgorithm::writeVirialEquilibrium(size_t i, std::ofstream &vtpFile) {
        vtpFile << virialEquilibrium[i] << '\n';
}

void nBodyAlgorithm::writeTotalEnergy(size_t i, std::ofstream &vtpFile) {
        vtpFile << totalEnergy[i] << '\n';

}

void nBodyAlgorithm::writePotentialEnergy(size_t i, std::ofstream &vtpFile) {
        vtpFile << potentialEnergy[i] << '\n';
}

void nBodyAlgorithm::writeKineticEnergy(size_t i, std::ofstream &vtpFile) {
        vtpFile << kineticEnergy[i] << '\n';
}

void nBodyAlgorithm::writeConnectivity(size_t i, std::ofstream &vtpFile) {
    for (std::size_t j = 0; j < positions_x[i].size(); ++j) {
        vtpFile << std::to_string(j) << ' ';
    }
}

void nBodyAlgorithm::writeOffsets(size_t i, std::ofstream &vtpFile) {
    for (std::size_t j = 1; j <= positions_x[i].size(); ++j) {
        vtpFile << std::to_string(j) << ' ';
    }
}

void nBodyAlgorithm::writeBodyClasses(const SimulationData &simulationData, std::ofstream &vtpFile) {
    for (const std::string &bodyClass: simulationData.body_classes) {
        vtpFile << 0 << '\n';
    }
}

void nBodyAlgorithm::writeNamesASCII(const SimulationData &simulationData, std::ofstream &vtpFile) const {
    for (const std::string &name: simulationData.names) {
        for (char c: name) {
            vtpFile << (int) c << ' '; // convert to ASCII
        }
        vtpFile << " 0" << '\n'; // add 0 to the end
    }
}

void nBodyAlgorithm::writeMasses(const SimulationData &simulationData, std::ofstream &vtpFile) const {
    for (double m: simulationData.mass) {
        vtpFile << m << '\n';
    }
}

void nBodyAlgorithm::writeAccelerations(size_t i, std::ofstream &vtpFile) {
    for (std::size_t j = 0; j < positions_x[i].size(); ++j) {
        vtpFile << acceleration[i][j] << '\n';
    }
}

void nBodyAlgorithm::writeVelocities(size_t i, std::ofstream &vtpFile) {
    for (std::size_t j = 0; j < velocities_x[i].size(); ++j) {
        vtpFile << velocities_x[i].at(j) << " "
                << velocities_y[i].at(j) << " "
                << velocities_z[i].at(j) << '\n';
    }
}

void nBodyAlgorithm::writeIDs(size_t i, std::ofstream &vtpFile) {
    for (std::size_t j = 0; j < positions_x[i].size(); ++j) {
        vtpFile << j << '\n';
    }
}

void nBodyAlgorithm::writePositions(size_t i, std::ofstream &vtpFile) {
    for (std::size_t j = 0; j < positions_x[i].size(); ++j) {
        vtpFile << positions_x[i].at(j) << " "
                << positions_y[i].at(j) << " "
                << positions_z[i].at(j) << '\n';
    }
}

void nBodyAlgorithm::writeFileHeader(size_t i, std::ofstream &vtpFile) {
    vtpFile << "<?xml version=\"1.0\"?>" << '\n'
            << R"(<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" header_type="UInt64">)" << '\n'
            << "<PolyData>" << '\n'
            << "<Piece NumberOfPoints=\"" << positions_x[i].size() << "\" NumberOfVerts=\"" << positions_x[i].size()
            << "\">" << '\n'
            << "<Points>" << '\n'
            << R"(<DataArray type="Float64" Name="position" NumberOfComponents="3" format="ascii">)" << '\n';
}

