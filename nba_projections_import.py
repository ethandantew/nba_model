from bs4 import BeautifulSoup
import requests
import pandas as pd
import json


def numFire():
    url = (
        "https://www.numberfire.com/nba/daily-fantasy/daily-basketball-projections#_=_"
    )
    r = requests.get(url)

    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table", {"class": "stat-table fixed-head"})
    table_body = table.find("tbody")
    trlist = []
    for tr in table_body.findAll("tr"):
        trlist.append(tr)

    players = []
    for row in trlist:
        for a in row.findAll("a", {"class": "full"}):
            playername = a.text.rstrip()
            playername = playername.lstrip()
            players.append(playername)

    pmins = []
    for row in trlist:
        for td in row.findAll("td", {"class": "min"}):
            pmin = td.text.rstrip()
            pmin = pmin.lstrip()
            pmins.append(pmin)

    pts = []
    for row in trlist:
        for td in row.findAll("td", {"class": "pts"}):
            ppts = td.text.rstrip()
            ppts = ppts.lstrip()
            pts.append(ppts)

    rebs = []
    for row in trlist:
        for td in row.findAll("td", {"class": "reb"}):
            prebs = td.text.rstrip()
            prebs = prebs.lstrip()
            rebs.append(prebs)

    asts = []
    for row in trlist:
        for td in row.findAll("td", {"class": "ast"}):
            past = td.text.rstrip()
            past = past.lstrip()
            asts.append(past)

    tpm = []
    for row in trlist:
        for td in row.findAll("td", {"class": "p3m"}):
            ptpm = td.text.rstrip()
            ptpm = ptpm.lstrip()
            tpm.append(ptpm)

    nfdf = pd.DataFrame(
        list(zip(players, pmins, pts, rebs, asts, tpm)),
        columns=["Player", "NF_Mins", "NF_Pts", "NF_Rebs", "NF_Asts", "NF_3PM"],
    )
    return nfdf


def sportsLine():
    url = "https://www.sportsline.com/nba/expert-projections/simulation/"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table", {"class": "sc-63394aef-7 ginGlq"})
    dfs = pd.read_html(str(table))
    df = dfs[0]
    cols = ["PLAYER", "POS", "TEAM", "PTS", "MIN", "AST", "TRB"]
    df = df[cols]
    colnames = ["PLAYER", "POS", "TEAM", "SL_PTS", "SL_MINS", "SL_ASTS", "SL_REBS"]
    df.columns = colnames
    return df


def main():
    numberfire_data = numFire()
    sportsline_data = sportsLine()

    merged_data = pd.merge(
        numberfire_data,
        sportsline_data,
        how="outer",
        left_on="Player",
        right_on="PLAYER",
    )
    df = merged_data
    df["Avg_mins"] = (df["NF_Mins"].astype(float) + df["SL_MINS"].astype(float)) / 2
    df["Avg_pts"] = (df["NF_Pts"].astype(float) + df["SL_PTS"].astype(float)) / 2
    df["Avg_rebs"] = (df["NF_Rebs"].astype(float) + df["SL_REBS"].astype(float)) / 2
    df["Avg_asts"] = (df["NF_Asts"].astype(float) + df["SL_ASTS"].astype(float)) / 2
    df["Mins_diff"] = df["NF_Mins"].astype(float) - df["SL_MINS"].astype(float)
    df["Pts_diff"] = df["NF_Pts"].astype(float) - df["SL_PTS"].astype(float)
    df["Rebs_diff"] = df["NF_Rebs"].astype(float) - df["SL_REBS"].astype(float)
    df["Asts_diff"] = df["NF_Asts"].astype(float) - df["SL_ASTS"].astype(float)
    df = df.dropna()
    cols = [
        "Player",
        "POS",
        "TEAM",
        "Avg_mins",
        "Avg_pts",
        "Avg_rebs",
        "Avg_asts",
        "NF_3PM",
        "Mins_diff",
        "Pts_diff",
        "Rebs_diff",
        "Asts_diff",
    ]
    df = df[cols]
    colnames = [
        "Player",
        "Position",
        "Team",
        "Min",
        "Pts",
        "Rebs",
        "Asts",
        "P3ms",
        "Mins_diff",
        "Pts_diff",
        "Rebs_diff",
        "Asts_diff",
    ]
    df.columns = colnames
    return df


def getLines():
    url = "https://rotogrinders.com/schedules/nba/dfs"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "lxml")
    scripts = soup.findAll("script")
    raw = str(scripts[-1])
    raw = raw.replace("\n", "")
    begin = raw.find("[")
    end = raw.rfind("}];")
    cut = raw[begin : end + 2]
    data = json.loads(cut)

    mykeys = ["team", "opponent", "line", "moneyline", "overunder", "projected"]
    masterdf = pd.DataFrame(
        columns=["team", "opponent", "line", "moneyline", "overunder", "projected"]
    )

    for i in range(len(data)):
        team = data[i]
        newdict = {}

        for key, value in team.items():
            if key in mykeys:
                newdict[key] = value

        df = pd.DataFrame(newdict, index=[0])
        masterdf = pd.concat([df, masterdf])

    masterdf["opponent"] = masterdf["opponent"].str.replace("@ ", "")
    masterdf["opponent"] = masterdf["opponent"].str.replace("vs. ", "")
    masterdf["opponent"] = masterdf["opponent"].str.lower()
    masterdf["team"] = masterdf["team"].str.lower()
    masterdf["team"] = masterdf["team"].str.replace("bkn", "brk")
    masterdf["team"] = masterdf["team"].str.replace("cha", "cho")
    masterdf["opponent"] = masterdf["opponent"].str.replace("cha", "cho")
    masterdf["opponent"] = masterdf["opponent"].str.replace("bkn", "brk")
    masterdf["line"] = pd.to_numeric(masterdf["line"])
    masterdf["projected"] = pd.to_numeric(masterdf["projected"])

    masterdf = masterdf[
        ["team", "opponent", "line", "moneyline", "overunder", "projected"]
    ]
    masterdf.columns = ["team", "opp", "line", "moneyline", "ou", "proj"]
    return masterdf


if __name__ == "__main__":
    main()
