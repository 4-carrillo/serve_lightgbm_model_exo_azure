import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from astropy.stats import LombScargle
import lightkurve as lk
from tqdm import tqdm

def extract_features(
    kicid, 
    mission='Kepler', 
    planet_star_radius_ratio=None, 
    a_by_rstar=None, 
    inclination_deg=None
):
    try:
        # Search for light curve based on mission
        lc_collection = lk.search_lightcurve(str(kicid), mission=mission, cadence='long').download_all()
        lc = lc_collection.stitch()

        flux = np.array(lc.flux.value)
        time = np.array(lc.time.value)

        # Basic flux statistics
        mean_flux = np.nanmean(flux)
        median_flux = np.nanmedian(flux)
        std_flux = np.nanstd(flux)
        skew_flux = skew(flux[~np.isnan(flux)])
        kurt_flux = kurtosis(flux[~np.isnan(flux)])
        min_flux = np.nanmin(flux)
        max_flux = np.nanmax(flux)
        transit_depth = median_flux - min_flux

        # Period detection using Lomb-Scargle
        valid = ~(np.isnan(time) | np.isnan(flux))
        frequency, power = LombScargle(time[valid], flux[valid]).autopower()
        period = 1 / frequency[np.argmax(power)]

        # Stellar metadata
        meta = lc.meta
        star_temp = meta.get('TEFF', np.nan)
        star_radius = meta.get('RADIUS', np.nan)
        star_metallicity = meta.get('FEH', np.nan)

        # Estimate planet radius
        if planet_star_radius_ratio is None:
            try:
                radius_ratio = np.sqrt(abs(transit_depth))
            except:
                radius_ratio = np.nan
        else:
            radius_ratio = planet_star_radius_ratio

        planetary_radius = (
            radius_ratio * star_radius 
            if star_radius and not np.isnan(star_radius) 
            else np.nan
        )

        # Impact parameter
        impact_parameter = np.nan
        if a_by_rstar is not None and inclination_deg is not None:
            inclination_rad = np.radians(inclination_deg)
            impact_parameter = a_by_rstar * np.cos(inclination_rad)

        # Pack results
        features = {
            'mean_flux': mean_flux,
            'median_flux': median_flux,
            'std_flux': std_flux,
            'skew_flux': skew_flux,
            'kurt_flux': kurt_flux,
            'min_flux': min_flux,
            'max_flux': max_flux,
            'transit_depth': transit_depth,
            'period': period,
            'star_temp': star_temp,
            'star_radius': star_radius,
            'star_metallicity': star_metallicity,
            'planetary_radius': planetary_radius,
            'impact_parameter': impact_parameter
        }
        return features
    
    except Exception as e:
        print(f"Error processing {mission} target ID {kicid}: {e}")
        return {k: np.nan for k in [
            'mean_flux', 'median_flux', 'std_flux', 'skew_flux', 'kurt_flux',
            'min_flux', 'max_flux', 'transit_depth', 'period', 'star_temp',
            'star_radius', 'star_metallicity', 'planetary_radius', 'impact_parameter'
        ]}

# --------------------------------------------------
# Batch processor (supports Kepler, K2, or TESS)
# --------------------------------------------------
def create_feature_dataset_in_batches(df, mission='Kepler', batch_size=50):
    all_features = []
    n = len(df)
    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        batch_df = df.iloc[start_idx:end_idx]
        batch_features = []

        for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Processing batch {start_idx}-{end_idx-1}"):
            kicid = str(row['kicid'])
            features = extract_features(
                kicid=kicid,
                mission=mission,
                planet_star_radius_ratio=row.get('planet_star_radius_ratio', None),
                a_by_rstar=row.get('a_by_rstar', None),
                inclination_deg=row.get('inclination_deg', None)
            )
            features['kicid'] = kicid
            if 'koi_disposition' in row:
                features['koi_disposition'] = row['koi_disposition']
            batch_features.append(features)

        batch_feature_df = pd.DataFrame(batch_features)
        all_features.append(batch_feature_df)

    feature_df = pd.concat(all_features, ignore_index=True)
    return feature_df
