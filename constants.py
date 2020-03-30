# ======================================== BOXES =============================================== #
ST_LOGO =   """<img src="https://media.licdn.com/dms/image/C4D0BAQF8YuUIurhzrw/company-logo_400_400/0?e=1574294400&v=beta&t=Qv5QXohlyU5kZzej8a6eJ3JoBdR2ML4iRaMpcSokA54" width="50" height="50"> """
ST_HEADER = f"""
         <center>
        {ST_LOGO}
        <font face = "Proxima Nova"> Research web app </font> 
        {ST_LOGO}
         </center>
    """
ST_HEADER_2 = f"""
         <center>
        {ST_LOGO}
        <font face = "Proxima Nova"> Upload new session </font> 
        {ST_LOGO}
         </center>
    """
TEXT_WIDTH = 220
CAT_WIDTH = 350
TYPE_WIDTH = 250
DEVICE_WIDTH = 200
SPACE_WIDTH = 1200
TEXT_HEIGHT = 30
BOX_HEIGHT = 50
TEXT_HEIGHT = 5
TXT_BOX_HEIGHT = 100
TXT_BOX_WIDTH = 600
BOX_WIDTH = 400

# ======================================== COLORS =============================================== #
BLUE_1 = '#41444b'
BLUE_2 = '#17c5cb'
BLUE_3 = '#51cb8c'
GREY = '#747777'
YELLOW = '#ffc43d'
OTHER_COLORS = ['#0000FF', '#0000A0', '#ADD8E6', '#800080', '#FFFF00', '#00FF00', '#FF00FF', '#FFFFFF', '#C0C0C0', '#808080', '#000000', '#FFA500', '#A52A2A']
PALETTE = [BLUE_1, BLUE_2, BLUE_3, GREY] + OTHER_COLORS

WEEK_DIC = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: "Saturday", 6: 'Sunday'}

WEEK_DIC_REVERSED = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, "Saturday": 5, 'Sunday': 6}


TIME_DIC = {'morning': 4, 'afternoon': 3, 'evening': 2, 'night': 1}
TIME_DIC_REVERSED = {4: 'morning', 3: 'afternoon', 2: 'evening', 1: 'night'}

TOOLS = "pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,reset,tap,save,box_select,"
