import ee
import pandas as pd
import numpy as np
import pdb

#June 2021 - Attempt to pull MODIS from EE

# ee.Authenticate()
ee.Initialize()

#load landsat image collection
lsatLST = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")

#load water mapping to mask non-water pixels
JRC = ee.Image("JRC/GSW1_3/GlobalSurfaceWater")

# (SIMON) "JRC is the frequency with which a pixel was classified as water since 1984 (excluding ice cover)
# a value of 90 gives us a reasonable water mask 
# If you want something more dynamic (varies as lakes shrink/grow), search for some of the MODIS water products in the
# EE catalogue
water_mask = JRC.select('occurrence').gt(90)

# Metadata filter goes here (dates, etc), with the MODIS collection there isn't much metadata
imgs = modisLST\
			.filter(ee.Filter.date('2000-01-01','2000-11-30'))

print(imgs.size().getInfo(), " images after filter")


# Water export function
def outWater(ft):

  # //Put a buffer around your site
	geo = ft.geometry().buffer(120);
  
  # //Filter the image collection to your site and map over all the overlapping images
	reduced = imgs.filterBounds(geo).map(function(img){
	    
	    # //Might be worth looking into the QA Bit band to see if you want to mask your image by quality
	    # //It would look something like.
	    # //var qa = img.select('QC_Day Bitmask').eq(0)
	    # // then below you'd add the line:
	    # //.updateMask(qa) under the updateMask(water)
	    
	    # //Mask out any non-water pixels and select bands to export, here we just doing temp but you could add more if want
		refOut = img.select('LST_Day_1km')
	    	.updateMask(water)
	    
	    # //This pulls an average of all pixel values within your buffered site. If you have questions on the arguments
	    # //you can look um up in the 'docs' tab to the left.
		ftOut = refOut.reduceRegion(ee.Reducer.mean(), geo, 30)
	    
	    # //This returns a feature sans geometry but with the summary band values
	    # //copyProperties will append any attributes of input feature
	    return ee.Feature(null,ftOut).copyProperties(ft).set('date',ee.Date(refOut.get('system:time_start')))
      })

	return reduced
}  
pdb.set_trace()
