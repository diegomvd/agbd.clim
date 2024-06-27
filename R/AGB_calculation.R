library("BIOMASS")
library("allodb")
library("dplyr")
library("tidyr")

agb_calculation <- function(tree_size_database, lat_col, lon_col){

  # Correction of genus and species names
  taxo <- correctTaxo(
    genus = tree_size_database$genus,
    species = tree_size_database$species
  )

  # Collecting wood density data according to species
  wd_data <- getWoodDensity(
    genus = taxo$genusCorrected,
    species = taxo$speciesCorrected,
    stand = tree_size_database$cluster
  )

  # Incorporating corrected genus, species and wood density data
  # in the database
  tree_size_database$speciesCorrected <- taxo$speciesCorrected
  tree_size_database$genusCorrected <- taxo$genusCorrected
  tree_size_database$meanWD <- wd_data$meanWD

  # Replace missing data values (-9999.9999) by NA
  tree_size_database[tree_size_database == -9999.9999] <- NA
  
  ############################################################################
  ## CALCULATION OF AGB WITH THE PANTROPICAL ALLOMETRY CHAVE ET AL. 2014 WITH
  ## THE BIOMASS PACKAGE
  ############################################################################

  # Calibrating height models at the cluster level to fill missing height data
  h_model <- modelHD(
    D = tree_size_database$stem_diameter_cm,
    H = tree_size_database$height_m,
    method = "log2",
    useWeight = TRUE,
    plot = tree_size_database$cluster
  )

  # Calculate heights according to the previously calibrated height model
  h_local <- retrieveH(
    D = tree_size_database$stem_diameter_cm,
    model = h_model,
    plot = tree_size_database$cluster
  )

  # Store results from height calculation in a new column
  tree_size_database$height_local_m <- h_local$H

  # Add height estimation data when measurements were missing
  tree_size_database <- transform(
    tree_size_database,
    height_final_m = ifelse(is.na(height_m), height_local_m, height_m)
  )

  # Compute AGB using the pan-tropical allometry
  tree_size_database$agb_tropical <- computeAGB(
    D = tree_size_database$stem_diameter_cm,
    WD = tree_size_database$meanWD,
    H = tree_size_database$height_final_m
  )

  # Transform AGB values from Mg to kg for coherence with the extra-tropical
  # estimation
  tree_size_database$agb_tropical <- 1000 * tree_size_database$agb_tropical

  ############################################################################
  ## CALCULATION OF AGB WITH EXTRATROPICAL ALLOMETRIES COMPILED IN THE ALLODB
  ## PACKAGE
  ############################################################################

  extra_tropical_agb <- get_biomass(
    dbh = tree_size_database$stem_diameter_cm,
    genus = tree_size_database$genusCorrected,
    species = tree_size_database$speciesCorrected,
    coords = tree_size_database[, c(lon_col, lat_col)]
  )

  tree_size_database$agb_extra_tropical <- extra_tropical_agb

  tree_size_database

}