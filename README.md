# JeepCodes 2.0

## Description

An assignment for **DIT837-Emerging Technologies 1**, taught by **Eugene C. Busico**.

A Python web app that finds jeepney routes within a user-specified fare budget. It compares travel time, fare, risk and transfers, shows alternatives and route graphs, and demonstrates disruptions and ML travel-time predictions using synthetic classroom data.

## Installation

Install Python 3.13. Keep `JeepCodes_2_AI_Route_Challenge.xlsx` in the parent folder of `app`, and retain the included `artifacts` folder.

Open PowerShell in the `app` folder and run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Usage

Start the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the local address shown in the terminal. Choose your origin, destination, preference, budget and scenario, then select **Find routes**. Explore route details, comparisons, activity and code lookup. Select **Reset to baseline** to return to normal conditions.
