.. cluster_generator documentation master file, created by
   sphinx-quickstart on Mon Jul 27 14:41:05 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



cluster_generator
=================
|yt-project| |docs| |testing| |Github Page| |Pylint| |coverage|

.. raw:: html

   <hr style="height:10px;background-color:black">

``cluster_generator`` is a cross-platform galaxy cluster initializer for N-body / hydrodynamics codes. ``cluster_generator`` supports
a variety of different possible configurations for the initialized galaxy clusters, including a variety of profiles, different construction
assumptions, and non-Newtonian gravity options.


.. raw:: html

   <hr style="color:black">

Features
========
- **Exhaustive versatility in available gravity theories and built-in profiles for a variety of purposes.**

.. raw:: html

   <table width="100%" table-layout="fixed">
   <tr>
      <td width="500" style="vertical-align: middle;">
      <h3 style="text-align: center;"> Gravities </h3> </td>
      <td width="500" style="vertical-align: middle;">
      <h3 style="text-align: center;"> Profiles </h3> </td>
   </tr>
   <tr>
      <td>
      <ul>
      <li> Newtonian Gravity </li>
      <li> <a href="https://en.wikipedia.org/wiki/AQUAL">AQUAL</a> (<a href="https://en.wikipedia.org/wiki/Modified_Newtonian_dynamics">MOND</a>ian) </li>
      <li> <a href="https://watermark.silverchair.com/mnras0403-0886.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA2UwggNhBgkqhkiG9w0BBwagggNSMIIDTgIBADCCA0cGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMGb-fef5Ctx7fV5WJAgEQgIIDGKepu4GZqp7A-i3x1gJbehOyxm4vG9kx4eohWE2ipnUGBf_25ORxOWVF3RG5-wVYger-KaprllV2wY4GHZ0wgwHvb21RfjhDFLkQH7iVLLR2PJTIIEXVfrdU1djeQtRmtcc-NbRF_iAAxoE6q3RDr3hhTndEaYnR_ElwbUhCctE9UcZHCqiD4-3MbwCfKmQm1NJRsI38vjiti9EoHbuz0VVT4-vyOrMIySssTS6A_qGUnW_r2Ar0yDBrtqbjJk5QkOhxG6ZtJQtLFWAJZ6rh5j66ifwBdmPpIaBlsPUM0FcctpFVi8BuvdhaQkE06WzsAvCm-etmIkzV83sNw0bT1G2l-YkZYMJ6IqqX8oqN4kzKxlwYp58CfHg4RNbIXtGwkwmw-FYIXRgbTlinbwlxa9pQO3XxtCySEjDbwFKGzQy-FtqNVDSWpAa4F87y1ie2XzU5pDZri7Fzw4Tw2W0izjptcb6hG1TPFFmQ_X-eXC48yToIXTaoVcdZrAiX3CtLWDLoXM2PbeaSs3ARJszpgZKavP3Et-kPnkhskV589iZSLKVGR4eR8uhCGXWu07sNFCixOMPA6KGkUOBrvukhhdcT0tjbX93SsPB_UH1MOyVowaKjJwkVGGUFEcb3LfYTsqBbs8PZWcu3Jomr6yd7zo5s6hExmHACfz_h_ic8kUZWSnAr3P2TlGNQgyX9DX9O6pghWMtkuhomWu4r9f6Mv2xMjVJ1A_ZCwGZIPm7SeBc70s1TaT4daMzLG6UDEQevzv8M3W7jkd4gYOWqBojvWz2JyR2SO7YWC_LHb4JD6VgrvsvAwZcrEoHyIGb_O25ULxEtgz2d8hd_cbmsO8XgE_VrTp2gz6Twp3c2J46_TpOJitrkKR7MUhr91MNHR5XypthZfQ5zxYR13fQ78TvE-RDe9enShgqlIYU0QQGmfSqocSx8LHFq8B1HPcQiCEMQl5-8tz39dANME-Hvmxn0a9XGblHeGeO5R6Dfgb-AyWW3oZJYJUmNHMpY2P-lS2Bpy8Fmhb_LthPyZqyqj7w2INBr6mWv2TkTpA">QUMOND</a> (<a href="https://en.wikipedia.org/wiki/Modified_Newtonian_dynamics">MOND</a>ian) </li>
      </ul>
      </td>
      <td>
      <ul>
      <li> NFW Profiles [Normal, Truncated, Super]  </li>
      <li> Hernquist Profiles [cored / uncored] </li>
      <li> Einasto Profile</li>
      <li> Power Law Profile</li>
      <li> Beta Model Profile</li>
      <li> Ascasibar & Markevitch (2006) [density, temperature]</li>
      <li> Vikhlinin (2006) [density,temperature]</li>
      <li> Entropy [baseline, broken, walker] </li>
      </ul>
      </td>
   </tr>
   </table>

- **Compatibility with many of the most common simulation softwares for pain-free simulation setup**
   - :ref:`RAMSES <ramses>`
   - :ref:`ATHENA++ <athena>`
   - :ref:`AREPO <arepo>`
   - :ref:`GAMER <gamer>`
   - :ref:`FLASH <flash>`
   - :ref:`GIZMO <gizmo>`
   - :ref:`ENZO <enzo>`

- **Magnetic field implementations**
- **Multiple virialization methods**

Getting Started
===============
.. raw:: html

   <hr style="height:10px;background-color:black">

To get started using the ``cluster_generator`` package, begin by following the installation instructions located on the :ref:`installation` page.
Once the package is installed, you can get started making clusters! We suggest first-time users check out our :ref:`quickstart` page, which has a load
of simple recipies to get your feet wet. Also check out the :ref:`examples` page for a more comprehensive set of examples for completing tasks in ``cluster_generator``.

Contents
========
.. raw:: html

   <hr style="height:10px;background-color:black">

.. toctree::
   :maxdepth: 1

   Getting_Started
   radial_profiles
   models
   particles
   fields
   initial_conditions
   codes
   examples
   api

Indices and tables
==================

.. raw:: html

   <hr style="height:10px;background-color:black">


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |yt-project| image:: https://img.shields.io/static/v1?label="works%20with"&message="yt"&color="blueviolet"
   :target: https://yt-project.org

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: https://eliza-diggins.github.io/cluster_generator/build/html/index.html

.. |testing| image:: https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/test.yml/badge.svg
.. |Pylint| image:: https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/pylint.yml/badge.svg
.. |Github Page| image:: https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/docs.yml/badge.svg
.. |coverage| image:: https://coveralls.io/repos/github/Eliza-Diggins/cluster_generator/badge.svg?branch=MOND
   :target: https://coveralls.io/github/Eliza-Diggins/cluster_generator?branch=MOND