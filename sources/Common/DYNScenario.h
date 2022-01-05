//
// Copyright (c) 2015-2021, RTE (http://www.rte-france.com)
// See AUTHORS.txt
// All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, you can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0
//
// This file is part of Dynawo, an hybrid C++/Modelica open source suite
// of simulation tools for power systems.
//

/**
 * @file  DYNScenario.h
 *
 * @brief task description : header file
 *
 */

#ifndef COMMON_DYNSCENARIO_H_
#define COMMON_DYNSCENARIO_H_

#include <string>

namespace DYNAlgorithms {
/**
 * @brief Scenario class
 * class to describe a scenario by its id and files
 */
class Scenario {
 public:
  /**
   * @brief set the id of the scenario
   * @param id id of the scenario
   */
  void setId(const std::string& id);

  /**
   * @brief set the dyd file to describe the scenario
   * @param file dyd file to describe the scenario
   */
  void setDydFile(const std::string& file);

  /**
   * @brief set the iidm file to describe the scenario
   * @param file iidm file to describe the scenario
   */
  void setIidmFile(const std::string& file);

  /**
   * @brief set the network par file to describe the scenario
   * @param file network par file to describe the scenario
   */
  void setNetworkParFile(const std::string& file);

  /**
   * @brief set the dyd file to describe the scenario
   * @param id network par id to describe the scenario
   */
  void setNetworkParId(const std::string& id);

  /**
   * @brief set the criteria file to use for the scenario
   * @param file criteria file to use for the scenario
   */
  void setCriteriaFile(const std::string& file);

  /**
   * @brief get the id of the scenario
   * @return id of the scenario
   */
  const std::string& getId() const;

  /**
   * @brief get the dyd file of the scenario
   * @return dyd file of the scenario
   */
  const std::string& getDydFile() const;

  /**
   * @brief get the iidm file of the scenario
   * @return iidm file of the scenario
   */
  const std::string& getIidmFile() const;

  /**
   * @brief get the network par file of the scenario
   * @return network par file of the scenario
   */
  const std::string& getNetworkParFile() const;

  /**
   * @brief get the network par id of the scenario
   * @return network par id of the scenario
   */
  const std::string& getNetworkParId() const;

  /**
   * @brief get the criteria file of the scenario
   * @return criteria file of the scenario
   */
  const std::string& getCriteriaFile() const;

 private:
  std::string id_;  ///< id of the scenario
  std::string dydFile_;  ///< dyd file to use for the scenario
  std::string iidmFile_;  ///< iidm file to use for the scenario
  std::string networkParFile_;  ///< network par file to use for the scenario
  std::string networkParId_;  ///< network par id to use for the scenario
  std::string criteriaFile_;  ///< criteria file to use for the scenario
};

}  // namespace DYNAlgorithms

#endif  // COMMON_DYNSCENARIO_H_
