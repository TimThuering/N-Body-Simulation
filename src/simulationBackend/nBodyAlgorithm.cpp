#include "nBodyAlgorithm.hpp"
#include <iostream>
#include <fstream>
#include <ctime>
#include <algorithm>
#include <filesystem>

void nBodyAlgorithm::generateParaViewOutput(const SimulationData &simulationData) {

    // create timestamp for directory name of output files
    std::time_t currentTime = std::time(nullptr);
    std::string time = static_cast<std::string>(std::ctime(&currentTime));
    std::replace(time.begin(), time.end(), ' ', '_');
    time = time.substr(0, time.size() - 1);

    std::string filePathBase =
            "N-Body-Simulation/output/" + time + '/';

    // create the directory for the output files
    std::filesystem::create_directory(filePathBase);

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
                << R"(group="" part="0" file=")" << time.append("/" + vtpFileName) << "\"/>";

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
    for (double VE: virialEquilibrium[i]) {
        vtpFile << VE << '\n';
    }
}

void nBodyAlgorithm::writeTotalEnergy(size_t i, std::ofstream &vtpFile) {
    for (double E: totalEnergy[i]) {
        vtpFile << E << '\n';
    }
}

void nBodyAlgorithm::writePotentialEnergy(size_t i, std::ofstream &vtpFile) {
    for (double E_pot: potentialEnergy[i]) {
        vtpFile << E_pot << '\n';
    }
}

void nBodyAlgorithm::writeKineticEnergy(size_t i, std::ofstream &vtpFile) {
    for (double E_kin: kineticEnergy[i]) {
        vtpFile << E_kin << '\n';
    }
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
    for (double acc: acceleration[i]) {
        vtpFile << acc << '\n';
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

