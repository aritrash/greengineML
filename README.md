Hello there!
This is a simple IoT - Embedded Systems project regarding how to detect artificial lighting (LED, Fluroscent or any other electrical / heat light source other than the sun) from natural lighting. This system finds its usage in automatic lighting and appliances management systems.
This uses <b> Logistic Regression </b> algorithms to take input from an LDR ADC values, and OpenWeatherAPI's Weather API to get exact sunset and sunrise times for our system to work and give a boolean output prediction of if there is any artificial light present.

Sadly I couldn't integrate the entire venv folder since this project was not directly Git posted from start. I request you to kindly set up venv by:<br>
<code>
  python -m venv venv
  ./venv/bin/activate
</code>
<br>
for Linux / CMD or <br>
<code>
  venv/Scripts/Activate.ps1
</code> <br>
for Windows Powershell, and then install all necessary libraries in the code. Inconvenience regretted.
