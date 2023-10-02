.. _qumond-gravity:
QUMOND Gravity
-----------------
.. raw:: html

   <hr style="height:5px;background-color:black">

.. raw:: html

    <style type="text/css">
    .tg  {border-collapse:collapse;border-color:#9ABAD9;border-spacing:0;}
    .tg td{background-color:#EBF5FF;border-color:#9ABAD9;border-style:solid;border-width:1px;color:#444;
      font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg th{background-color:#409cff;border-color:#9ABAD9;border-style:solid;border-width:1px;color:#fff;
      font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg .tg-sabo{background-color:#ebf5ff;color:#000000;text-align:left;vertical-align:top}
    .tg .tg-0blg{color:#ffffff;font-weight:bold;text-align:left;vertical-align:top}
    .tg .tg-ye1f{font-family:"Lucida Console", Monaco, monospace !important;text-align:left;vertical-align:top}
    .tg .tg-dg13{background-color:#409cff;color:#ffffff;font-weight:bold;text-align:left;vertical-align:top}
    .tg .tg-0lax{text-align:left;vertical-align:top}
    </style>
    <table class="tg">
    <thead>
      <tr>
        <th class="tg-0blg">Name</th>
        <th class="tg-sabo" colspan="5">Newtonian Gravity</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="tg-dg13">Class</td>
        <td class="tg-ye1f" colspan="5">gravity.NewtonianGravity</td>
      </tr>
      <tr>
        <td class="tg-dg13">Type</td>
        <td class="tg-0lax" colspan="5">Classical</td>
      </tr>
      <tr>
        <td class="tg-dg13">Available Potential Solvers</td>
        <td class="tg-dg13">Name</td>
        <td class="tg-dg13">Method</td>
        <td class="tg-dg13" colspan="2">Description</td>
        <td class="tg-dg13">Rank</td>
      </tr>
      <tr>
        <td class="tg-0lax"></td>
        <td class="tg-0lax">GF Solver</td>
        <td class="tg-ye1f">NewtonianGravity._calcpo_gf</td>
        <td class="tg-0lax" colspan="2">Solves the potential by integrating from known acceleration. <br>Applies zero potential boundary at infinity.</td>
        <td class="tg-0lax">1</td>
      </tr>
      <tr>
        <td class="tg-0lax"></td>
        <td class="tg-0lax">DRM Solver</td>
        <td class="tg-ye1f">NewtonianGravity._calcpo_drm</td>
        <td class="tg-0lax" colspan="2">Solves the potential using the total density and mass profiles. <br>Solver uses spherical shells within and without each radius. Boundary<br>is solved for analytically with value zero at the infinite boundary.</td>
        <td class="tg-0lax">2</td>
      </tr>
    </tbody>
    </table>

.. raw:: html

   <hr style="height:5px;background-color:black">

.. warning::

    Uh Oh! This documentation hasn't been written yet. We'll get to it as soon as we can!


Description
+++++++++++

Methods
+++++++

Notes
+++++

References
++++++++++