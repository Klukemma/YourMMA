"""How the app names and colours a club.

football-data.co.uk's own names ("Man United", "Nott'm Forest", "Ath
Madrid") are the engine's keys and never change; this is display only:
the name a fan would write, a three-letter code for the badge chip, and
the club's colours (chip background, chip text). A club missing here is
shown under its football-data.co.uk name with a neutral chip - never a
guessed identity.
"""

# key: (display name, code, background, text)
TEAMS = {
    # England
    "Arsenal": ("Arsenal", "ARS", "#EF0107", "#FFFFFF"),
    "Aston Villa": ("Aston Villa", "AVL", "#670E36", "#95BFE5"),
    "Bournemouth": ("Bournemouth", "BOU", "#DA291C", "#000000"),
    "Brentford": ("Brentford", "BRE", "#E30613", "#FFFFFF"),
    "Brighton": ("Brighton", "BHA", "#0057B8", "#FFFFFF"),
    "Burnley": ("Burnley", "BUR", "#6C1D45", "#99D6EA"),
    "Cardiff": ("Cardiff City", "CAR", "#0070B5", "#FFFFFF"),
    "Chelsea": ("Chelsea", "CHE", "#034694", "#FFFFFF"),
    "Coventry": ("Coventry City", "COV", "#6CABDD", "#FFFFFF"),
    "Crystal Palace": ("Crystal Palace", "CRY", "#1B458F", "#C4122E"),
    "Everton": ("Everton", "EVE", "#003399", "#FFFFFF"),
    "Fulham": ("Fulham", "FUL", "#FFFFFF", "#000000"),
    "Huddersfield": ("Huddersfield Town", "HUD", "#0E63AD", "#FFFFFF"),
    "Hull": ("Hull City", "HUL", "#F5A12D", "#000000"),
    "Ipswich": ("Ipswich Town", "IPS", "#0044A9", "#FFFFFF"),
    "Leeds": ("Leeds United", "LEE", "#FFFFFF", "#1D428A"),
    "Leicester": ("Leicester City", "LEI", "#003090", "#FDBE11"),
    "Liverpool": ("Liverpool", "LIV", "#C8102E", "#FFFFFF"),
    "Luton": ("Luton Town", "LUT", "#F78F1E", "#002D62"),
    "Man City": ("Manchester City", "MCI", "#6CABDD", "#1C2C5B"),
    "Man United": ("Manchester United", "MUN", "#DA291C", "#FBE122"),
    "Middlesbrough": ("Middlesbrough", "MID", "#E11B22", "#FFFFFF"),
    "Newcastle": ("Newcastle United", "NEW", "#241F20", "#FFFFFF"),
    "Norwich": ("Norwich City", "NOR", "#FFF200", "#00A650"),
    "Nott'm Forest": ("Nottingham Forest", "NFO", "#DD0000", "#FFFFFF"),
    "Sheffield United": ("Sheffield United", "SHU", "#EE2737", "#FFFFFF"),
    "Southampton": ("Southampton", "SOU", "#D71920", "#FFFFFF"),
    "Stoke": ("Stoke City", "STK", "#E03A3E", "#FFFFFF"),
    "Sunderland": ("Sunderland", "SUN", "#EB172B", "#FFFFFF"),
    "Swansea": ("Swansea City", "SWA", "#FFFFFF", "#121212"),
    "Tottenham": ("Tottenham Hotspur", "TOT", "#132257", "#FFFFFF"),
    "Watford": ("Watford", "WAT", "#FBEE23", "#ED2127"),
    "West Brom": ("West Bromwich Albion", "WBA", "#122F67", "#FFFFFF"),
    "West Ham": ("West Ham United", "WHU", "#7A263A", "#1BB1E7"),
    "Wolves": ("Wolverhampton", "WOL", "#FDB913", "#231F20"),
    # Spain
    "Alaves": ("Alavés", "ALA", "#0761AF", "#FFFFFF"),
    "Almeria": ("Almería", "ALM", "#EE1119", "#FFFFFF"),
    "Ath Bilbao": ("Athletic Club", "ATH", "#EE2523", "#FFFFFF"),
    "Ath Madrid": ("Atlético Madrid", "ATM", "#CB3524", "#FFFFFF"),
    "Barcelona": ("Barcelona", "BAR", "#A50044", "#EDBB00"),
    "Betis": ("Real Betis", "BET", "#00954C", "#FFFFFF"),
    "Cadiz": ("Cádiz", "CAD", "#FDE607", "#0045A7"),
    "Celta": ("Celta Vigo", "CEL", "#8AC3EE", "#FFFFFF"),
    "Eibar": ("Eibar", "EIB", "#003C7E", "#C8102E"),
    "Elche": ("Elche", "ELC", "#05642C", "#FFFFFF"),
    "Espanol": ("Espanyol", "ESP", "#007FC8", "#FFFFFF"),
    "Getafe": ("Getafe", "GET", "#005999", "#FFFFFF"),
    "Girona": ("Girona", "GIR", "#CD2534", "#FFFFFF"),
    "Granada": ("Granada", "GRA", "#C31632", "#FFFFFF"),
    "Huesca": ("Huesca", "HUE", "#1E3F8C", "#B5121B"),
    "La Coruna": ("Deportivo La Coruña", "DEP", "#1B5EAB", "#FFFFFF"),
    "Las Palmas": ("Las Palmas", "LPA", "#FFE400", "#0055A5"),
    "Leganes": ("Leganés", "LEG", "#0F4BA1", "#FFFFFF"),
    "Levante": ("Levante", "LEV", "#B4053F", "#004D98"),
    "Malaga": ("Málaga", "MAL", "#0E74B9", "#FFFFFF"),
    "Mallorca": ("Mallorca", "MLL", "#E20613", "#000000"),
    "Osasuna": ("Osasuna", "OSA", "#D91A21", "#0A346F"),
    "Oviedo": ("Real Oviedo", "OVI", "#0058A8", "#FFFFFF"),
    "Real Madrid": ("Real Madrid", "RMA", "#FFFFFF", "#00529F"),
    "Santander": ("Racing Santander", "RAC", "#008D36", "#FFFFFF"),
    "Sevilla": ("Sevilla", "SEV", "#D81E05", "#FFFFFF"),
    "Sociedad": ("Real Sociedad", "RSO", "#0067B1", "#FFFFFF"),
    "Sp Gijon": ("Sporting Gijón", "SPO", "#D31F26", "#FFFFFF"),
    "Valencia": ("Valencia", "VAL", "#FFFFFF", "#000000"),
    "Valladolid": ("Real Valladolid", "VLL", "#5B2B82", "#FFFFFF"),
    "Vallecano": ("Rayo Vallecano", "RAY", "#E53027", "#FFFFFF"),
    "Villarreal": ("Villarreal", "VIL", "#FFE667", "#005187"),
    # Italy
    "Atalanta": ("Atalanta", "ATA", "#1E71B8", "#000000"),
    "Benevento": ("Benevento", "BEN", "#D7141A", "#FFDD00"),
    "Bologna": ("Bologna", "BOL", "#1A2F48", "#A21C26"),
    "Brescia": ("Brescia", "BRE", "#004A99", "#FFFFFF"),
    "Cagliari": ("Cagliari", "CAG", "#002350", "#A6192E"),
    "Chievo": ("Chievo", "CHI", "#FFDD00", "#002A5C"),
    "Como": ("Como", "COM", "#003DA5", "#FFFFFF"),
    "Cremonese": ("Cremonese", "CRE", "#A0141E", "#808080"),
    "Crotone": ("Crotone", "CRO", "#004C97", "#E20613"),
    "Empoli": ("Empoli", "EMP", "#00579C", "#FFFFFF"),
    "Fiorentina": ("Fiorentina", "FIO", "#482E92", "#FFFFFF"),
    "Frosinone": ("Frosinone", "FRO", "#FFDD00", "#0055A4"),
    "Genoa": ("Genoa", "GEN", "#A41F35", "#002A5C"),
    "Inter": ("Inter", "INT", "#0068A8", "#000000"),
    "Juventus": ("Juventus", "JUV", "#000000", "#FFFFFF"),
    "Lazio": ("Lazio", "LAZ", "#87D8F7", "#15366F"),
    "Lecce": ("Lecce", "LEC", "#FFDD00", "#D7141A"),
    "Milan": ("Milan", "MIL", "#FB090B", "#000000"),
    "Monza": ("Monza", "MON", "#E2001A", "#FFFFFF"),
    "Napoli": ("Napoli", "NAP", "#12A0D7", "#FFFFFF"),
    "Palermo": ("Palermo", "PAL", "#F4A6C9", "#000000"),
    "Parma": ("Parma", "PAR", "#FFDD00", "#1B4593"),
    "Pescara": ("Pescara", "PES", "#2D8FD5", "#FFFFFF"),
    "Pisa": ("Pisa", "PIS", "#0C2A5B", "#000000"),
    "Roma": ("Roma", "ROM", "#8E1F2F", "#F0BC42"),
    "Salernitana": ("Salernitana", "SAL", "#7A1F35", "#FFFFFF"),
    "Sampdoria": ("Sampdoria", "SAM", "#1B5497", "#FFFFFF"),
    "Sassuolo": ("Sassuolo", "SAS", "#00A752", "#000000"),
    "Spal": ("SPAL", "SPA", "#FFFFFF", "#005AA9"),
    "Spezia": ("Spezia", "SPE", "#FFFFFF", "#000000"),
    "Torino": ("Torino", "TOR", "#8A1E03", "#FFFFFF"),
    "Udinese": ("Udinese", "UDI", "#000000", "#FFFFFF"),
    "Venezia": ("Venezia", "VEN", "#000000", "#F37321"),
    "Verona": ("Hellas Verona", "VER", "#002F6C", "#FFDD00"),
    # Germany
    "Augsburg": ("Augsburg", "FCA", "#BA3733", "#FFFFFF"),
    "Bayern Munich": ("Bayern Munich", "FCB", "#DC052D", "#FFFFFF"),
    "Bielefeld": ("Arminia Bielefeld", "DSC", "#004E95", "#FFFFFF"),
    "Bochum": ("Bochum", "BOC", "#005CA9", "#FFFFFF"),
    "Darmstadt": ("Darmstadt", "D98", "#004E9E", "#FFFFFF"),
    "Dortmund": ("Dortmund", "BVB", "#FDE100", "#000000"),
    "Ein Frankfurt": ("Eintracht Frankfurt", "SGE", "#E1000F", "#000000"),
    "Elversberg": ("Elversberg", "SVE", "#000000", "#FFFFFF"),
    "FC Koln": ("Köln", "KOE", "#ED1C24", "#FFFFFF"),
    "Fortuna Dusseldorf": ("Fortuna Düsseldorf", "F95", "#DA251D", "#FFFFFF"),
    "Freiburg": ("Freiburg", "SCF", "#000000", "#FFFFFF"),
    "Greuther Furth": ("Greuther Fürth", "SGF", "#009A44", "#FFFFFF"),
    "Hamburg": ("Hamburg", "HSV", "#0A3F86", "#FFFFFF"),
    "Hannover": ("Hannover 96", "H96", "#00A651", "#000000"),
    "Heidenheim": ("Heidenheim", "FCH", "#E2001A", "#003B79"),
    "Hertha": ("Hertha Berlin", "BSC", "#005CA9", "#FFFFFF"),
    "Hoffenheim": ("Hoffenheim", "TSG", "#1961B5", "#FFFFFF"),
    "Holstein Kiel": ("Holstein Kiel", "KSV", "#003F8A", "#FFFFFF"),
    "Ingolstadt": ("Ingolstadt", "FCI", "#000000", "#E2001A"),
    "Leverkusen": ("Leverkusen", "B04", "#E32221", "#000000"),
    "M'gladbach": ("Mönchengladbach", "BMG", "#000000", "#FFFFFF"),
    "Mainz": ("Mainz", "M05", "#C3141E", "#FFFFFF"),
    "Nurnberg": ("Nürnberg", "FCN", "#AD1732", "#FFFFFF"),
    "Paderborn": ("Paderborn", "SCP", "#0058A3", "#FFFFFF"),
    "RB Leipzig": ("RB Leipzig", "RBL", "#FFFFFF", "#DD0741"),
    "Schalke 04": ("Schalke", "S04", "#004D9D", "#FFFFFF"),
    "St Pauli": ("St. Pauli", "STP", "#624839", "#FFFFFF"),
    "Stuttgart": ("Stuttgart", "VFB", "#FFFFFF", "#E32219"),
    "Union Berlin": ("Union Berlin", "FCU", "#EB1923", "#FFFFFF"),
    "Werder Bremen": ("Werder Bremen", "SVW", "#1D9053", "#FFFFFF"),
    "Wolfsburg": ("Wolfsburg", "WOB", "#65B32E", "#FFFFFF"),
    # France
    "Ajaccio": ("Ajaccio", "ACA", "#E2001A", "#FFFFFF"),
    "Amiens": ("Amiens", "ASC", "#FFFFFF", "#000000"),
    "Angers": ("Angers", "ANG", "#000000", "#FFFFFF"),
    "Auxerre": ("Auxerre", "AJA", "#FFFFFF", "#005BAC"),
    "Bastia": ("Bastia", "BAS", "#0055A4", "#FFFFFF"),
    "Bordeaux": ("Bordeaux", "BOR", "#001B50", "#FFFFFF"),
    "Brest": ("Brest", "BRE", "#E30613", "#FFFFFF"),
    "Caen": ("Caen", "SMC", "#004B93", "#E30613"),
    "Clermont": ("Clermont", "CF6", "#C8102E", "#002D62"),
    "Dijon": ("Dijon", "DFC", "#E2001A", "#FFFFFF"),
    "Guingamp": ("Guingamp", "EAG", "#E2001A", "#000000"),
    "Le Havre": ("Le Havre", "HAC", "#7FB4E3", "#00305E"),
    "Le Mans": ("Le Mans", "LMF", "#E2001A", "#FFDD00"),
    "Lens": ("Lens", "RCL", "#FFDD00", "#E2001A"),
    "Lille": ("Lille", "LIL", "#E01E13", "#FFFFFF"),
    "Lorient": ("Lorient", "FCL", "#F58113", "#000000"),
    "Lyon": ("Lyon", "LYO", "#FFFFFF", "#1A3D8F"),
    "Marseille": ("Marseille", "MAR", "#2FAEE0", "#FFFFFF"),
    "Metz": ("Metz", "FCM", "#7A1E3A", "#FFFFFF"),
    "Monaco": ("Monaco", "ASM", "#E2001A", "#FFFFFF"),
    "Montpellier": ("Montpellier", "MTP", "#002F6C", "#F58113"),
    "Nancy": ("Nancy", "NAN", "#FFFFFF", "#E2001A"),
    "Nantes": ("Nantes", "FCN", "#FCD405", "#00843D"),
    "Nice": ("Nice", "NIC", "#C8102E", "#000000"),
    "Nimes": ("Nîmes", "NIM", "#E2001A", "#FFFFFF"),
    "Paris FC": ("Paris FC", "PFC", "#1D2E5C", "#FFFFFF"),
    "Paris SG": ("Paris Saint-Germain", "PSG", "#004170", "#DA291C"),
    "Reims": ("Reims", "SDR", "#E2001A", "#FFFFFF"),
    "Rennes": ("Rennes", "REN", "#E2001A", "#000000"),
    "St Etienne": ("Saint-Étienne", "STE", "#009639", "#FFFFFF"),
    "Strasbourg": ("Strasbourg", "STR", "#009FE3", "#FFFFFF"),
    "Toulouse": ("Toulouse", "TFC", "#5B2A86", "#FFFFFF"),
    "Troyes": ("Troyes", "TRO", "#005BAC", "#FFFFFF"),
}

NEUTRAL = ("#5B6170", "#FFFFFF")
# The badge's code is three bold capitals: held to the 3:1 WCAG asks of large
# text and UI glyphs, which keeps a club's own white-on-red (Arsenal, 4.5:1)
# and replaces only the clashes (Crystal Palace's red on blue, 1.5:1).
MIN_CONTRAST = 3.0


def _luminance(hex_colour):
    h = hex_colour.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b):
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def readable(bg, fg):
    """The club's own text colour where it reads on its background; white
    or black, whichever reads better, where it does not (Crystal Palace's
    red on blue, say)."""
    if contrast(bg, fg) >= MIN_CONTRAST:
        return fg
    return max(("#FFFFFF", "#000000"), key=lambda c: contrast(bg, c))


def display(key):
    """{name, code, bg, fg} for a football-data.co.uk team name."""
    if key in TEAMS:
        name, code, bg, fg = TEAMS[key]
    else:
        letters = "".join(ch for ch in str(key).upper() if ch.isalpha())
        name, code, (bg, fg) = key, letters[:3] or "?", NEUTRAL
    return {"name": name, "code": code, "bg": bg, "fg": readable(bg, fg)}
