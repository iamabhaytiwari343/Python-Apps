# Import QRCode from pyqrcode 
# package for the qr code generator
import pyqrcode 
import png 
from pyqrcode import QRCode 
  
  
# String which represents the QR code 
s = "https://www.youtube.com/@Explorewithsonu_07" 
  
# Generate QR code 
url = pyqrcode.create(s) 
  
# Create and save the svg file naming "myqr.svg" 
url.svg("myqr.svg", scale = 8) 
  
# Create and save the png file naming "myqr.png" 
url.png('myqr.png', scale = 6)