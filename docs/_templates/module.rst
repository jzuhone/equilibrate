{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :toctree:
      :recursive:

   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block modules %}
   {% if modules %}
   .. rubric:: Modules

   .. autosummary::
      :toctree:
      :template: modules.rst
      :recursive:
   {% for item in modules %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :toctree:
      :template: class.rst
      :recursive:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
