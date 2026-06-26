#define _CRT_SECURE_NO_WARNINGS
#include "functions.h"
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

int main() {
    std::cout << "Scegli modalita' principale:" << std::endl;
    std::cout << "1 = funzionamento normale, senza calcolare la sensitivity" << std::endl;
    std::cout << "2 = calcola la sensitivity su una cartella di immagini" << std::endl;
    std::cout << "3 = estrai tutte le ROI trovate e calcola tutte le feature " << std::endl;
    std::cout << "Premi 1, 2 oppure 3 e poi INVIO: ";

    std::string mainMode;
    std::getline(std::cin, mainMode);

    try {
        if (!mainMode.empty() && mainMode[0] == '2') {
            runSensitivityMode();
        }
        else if (!mainMode.empty() && mainMode[0] == '3') {
            runRoiFeatureExtractionMode();
        }
        else {
            runNormalModeWithoutSensitivity();
        }
    }
    catch (const std::exception& e) {
        std::cout << std::endl;
        std::cout << "Errore: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cout << std::endl;
        std::cout << "Errore sconosciuto." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
