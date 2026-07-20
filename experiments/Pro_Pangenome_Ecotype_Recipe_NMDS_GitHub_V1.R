## Necessary library packages
library(dplyr)
library(tidyr)
library(ggplot2)
library(tibble)
library(vegan)
library(stringr)

df <- read.csv("file/path/for Grump dataset/grump_asv_long_version-1.3.5.csv")

df$Full_Taxonomy <- paste(df$Domain, df$Supergroup, df$Division, df$Phylum, df$Class,
                          df$Order, df$Family, df$Genus, df$Species, df$ProPortal_ASV_Ecotype, sep = ";")

## G1 - Subset all Prochlorococcus and identify ecological regimes in the upper 200m
df.sub <- df %>%
  filter(Eco_relevant_plank_groups=="Prochlorococcus",
         depth < 201)
unique(df.sub$Eco_relevant_plank_groups)
unique(df.sub$ProPortal_ASV_Ecotype[grep("Prochlorococcus", df.sub$Full_Taxonomy)])

## Reannotation of Pro using the ProSynTaxDB
pro.ann <- read.csv("filepath/Prochlorococcus_asv_clade_assignments_SB_v1.csv")
df.sub <- df.sub %>%
  left_join(
    pro.ann %>%
      select(ASV_hash, ProSynTax = Clade),
    by = "ASV_hash"
  )
unique(df.sub$ProSynTax)

## Breakdown of samples coverage of Prochlorococcus
total_reads <- df %>%
  filter(SampleID %in% df.sub$SampleID) %>%
  group_by(SampleID) %>%
  summarise(Total_reads = sum(Corrected_Sequence_Counts, na.rm = TRUE),
            .groups = "drop")

pro_reads <- df.sub %>%
  group_by(SampleID) %>%
  summarise(Pro_reads = sum(Corrected_Sequence_Counts, na.rm = TRUE),
            .groups = "drop")

pro_rel <- total_reads %>%
  left_join(pro_reads, by = "SampleID") %>%
  mutate(Pro_reads = ifelse(is.na(Pro_reads), 0, Pro_reads),
         Pro_rel_abundance = Pro_reads / Total_reads)

hist_pro <- ggplot(pro_rel, aes(x = Pro_rel_abundance)) +
  geom_histogram(bins = 50, fill = "gray70", color = "black") +
  scale_x_log10() +
  geom_vline(xintercept = c(0.001, 0.01, 0.1),
             linetype = "dashed",
             color = c("red", "orange", "blue")) +
  annotate("text", x = 0.001, y = Inf, label = "0.1%", vjust = 2, color = "red") +
  annotate("text", x = 0.01, y = Inf, label = "1%", vjust = 2, color = "orange") +
  annotate("text", x = 0.1, y = Inf, label = "10%", vjust = 2, color = "blue") +
  labs(x = "Prochlorococcus relative abundance (log10)",
       title = "Choosing a minimum Prochlorococcus threshold") +
  theme_bw()
hist_pro

## Create ecological bins
sample_bins <- df.sub %>%
  select(
    SampleID,
    depth,
    Longhurst_Long
  ) %>%
  distinct() %>%
  mutate(
    region_group = case_when(
      grepl("Coastal", Longhurst_Long) ~ "Coastal",
      grepl("Trades", Longhurst_Long) ~ "Trades",
      grepl("Westerlies", Longhurst_Long) ~ "Westerlies",
      grepl("Polar", Longhurst_Long) ~ "Polar",
      TRUE ~ "Other"
    ),
    
    depth_bin = cut(
      depth,
      breaks = c(0, 50, 100, 200),
      labels = c("upper50m", "50to100m", "100to200m"),
      include.lowest = TRUE
    ),
    
    manual.bin = paste(region_group, depth_bin, sep = "_")
  )

## Add Pro % RA information
sample_bins <- sample_bins %>%
  left_join(
    pro_rel %>%
      select(SampleID, Pro_rel_abundance),
    by = "SampleID"
  ) %>%
  mutate(
    Pro_group = ifelse(
      Pro_rel_abundance > 0.001,
      ">0.1%",
      "<=0.1%"
    )
  )

## Summary table 
bin_summary <- sample_bins %>%
  group_by(manual.bin, Pro_group) %>%
  summarise(
    n_samples = n_distinct(SampleID),
    .groups = "drop"
  ) %>%
  tidyr::pivot_wider(
    names_from = Pro_group,
    values_from = n_samples,
    values_fill = 0
  ) %>%
  mutate(
    Total = `>0.1%` + `<=0.1%`,
    Percent_retained = round(100 * `>0.1%` / Total, 1)
  ) %>%
  arrange(manual.bin)

bin_summary

ggplot(pro_rel %>% filter(Pro_rel_abundance > 0.001),
       aes(x = Pro_reads)) +
  geom_histogram(bins = 50, fill = "gray70", color = "black") +
  scale_x_log10() +
  labs(x = "Prochlorococcus reads (log10)",
       title = "Distribution of total Prochlorococcus reads in all samples >0.1% of total reads") +
  theme_bw()

## Move forward with samples where Pro contributes greater than 0.1%
df.sub.v2 <- df.sub %>%
  filter(SampleID %in% (pro_rel %>%
                          filter(Pro_rel_abundance > 0.001) %>%
                          pull(SampleID)))

## Recipe plots
df.sub.v2 <- df.sub.v2 %>%
  left_join(
    sample_bins %>%
      select(SampleID, manual.bin),
    by = "SampleID"
  ) %>%
  ## REMOVE coastal deep samples early
  filter(manual.bin != "Coastal_100to200m")

ecotype_recipe <- df.sub.v2 %>%
  group_by(manual.bin, ProSynTax) %>%
  summarise(
    Abundance = sum(Corrected_Sequence_Counts, na.rm = TRUE),
    .groups = "drop"
  )

ecotype_recipe$manual.bin <- factor(
  ecotype_recipe$manual.bin,
  levels = c(
    "Coastal_upper50m",
    "Coastal_50to100m",
    "Trades_upper50m",
    "Trades_50to100m",
    "Trades_100to200m",
    "Westerlies_upper50m",
    "Westerlies_50to100m",
    "Westerlies_100to200m"
  )
)

sample_counts <- df.sub.v2 %>%
  distinct(SampleID, manual.bin) %>%
  count(manual.bin, name = "n")

ecotype_recipe <- ecotype_recipe %>%
  group_by(manual.bin) %>%
  mutate(RelAbundance = Abundance / sum(Abundance)) %>%
  ungroup()

## determine which clades to depict
ecotype_recipe %>%
  group_by(ProSynTax) %>%
  summarise(
    max_RA = max(RelAbundance),
    mean_RA = mean(RelAbundance),
    bins_present = sum(RelAbundance > 0.01)
  ) %>%
  arrange(desc(max_RA))

## Add new plotting merger
ecotype_recipe <- ecotype_recipe %>%
  mutate(
    ProSynTax_plot = case_when(
      ProSynTax %in% c("AMZ-I", "AMZ-III") ~ "AMZ",
      ProSynTax %in% c("HLIV", "HLII-HLVI", "LLVIII") ~ "Other",
      ProSynTax == "LLII-LLIII" ~ "LLII-III",
      ProSynTax == "unclassified" ~ "Unclassified",
      TRUE ~ ProSynTax
    )
  )

p.biome <- ggplot(ecotype_recipe,
                  aes(x = manual.bin,
                      y = RelAbundance,
                      fill = ProSynTax_plot)) +
  
  geom_bar(stat = "identity") +
  
  geom_text(
    data = sample_counts,
    aes(x = manual.bin, y = 1.03, label = paste0("n=", n)),
    inherit.aes = FALSE,
    size = 4
  ) +
  
  scale_fill_manual(values = c(
    "AMZ" = "#66a61e",
    "HLI" = "#cb181d",
    "HLII" = "#fd8d3c",
    "HLIII" = "#8c510a",
    "HLVI" = "#dfc27d",
    "LLI" = "#08519c",
    "LLII-III" = "#54278f",
    "LLIV" = "#7bccc4",
    "LLVII" = "#8da0cb",
    "Other" = "#e7298a",
    "Unclassified" = "#999999"
  ), name = "Ecotype Clades") +
  
  scale_y_continuous(
    breaks = seq(0, 1, by = 0.2),
    limits = c(0, 1.08)
  ) +
  
  labs(
    title = "Prochlorococcus Ecotype Clades Across Depth-Biome Partitions",
    x = "Depth-Biome Partitions",
    y = "Relative Abundance"
  ) +
  
  theme_minimal(base_size = 14) +
  
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.major.x = element_blank()
  )

p.biome

ggsave(plot = p.biome, filename = "_______.pdf",
       path = "file/path", 
       width = 12, height = 8)

## Save the % dataframe for reanalysis and plotting
save.df <- ecotype_recipe[,c("manual.bin", "ProSynTax", "RelAbundance", "ProSynTax_plot")]
setwd("file/path")
write.csv(save.df, file = "Pro Ecotype Recipe Rel Abundance 8 Manual Bins Final.csv", row.names = F)

## ASV matrix (no aggregation into ecotype clades necessary based on prior sensitivity analysis)
asv_mat.v2 <- df.sub.v2 %>%
  group_by(SampleID, ASV_hash) %>%
  summarise(Abundance = sum(Corrected_Sequence_Counts, na.rm = TRUE),
            .groups = "drop") %>%
  pivot_wider(names_from = ASV_hash,
              values_from = Abundance,
              values_fill = 0) %>%
  column_to_rownames("SampleID") %>%
  as.matrix()

# ## Relative abundance transform
asv_mat_rel.v2 <- decostand(asv_mat.v2, method = "total")

## Bray-Curtis distance
bray.v2 <- vegdist(asv_mat_rel.v2, method = "bray")

## Hierarchical clustering
hc.v2 <- hclust(bray.v2, method = "ward.D2")
plot(hc.v2, main = "ASV-level clustering")
rect.hclust(hc.v2, k = 6, border = 2:7)
## NMDS
nmds.v2 <- metaMDS(asv_mat_rel.v2,
                   distance = "bray",
                   k = 2,
                   trymax = 20)

scores_df.v2 <- as.data.frame(scores(nmds.v2, display = "sites"))
scores_df.v2$Sample <- rownames(scores_df.v2)

## Attach metadata
meta.v2 <- df.sub.v2 %>%
  select(SampleID, depth, Longhurst_Long, manual.bin) %>%
  distinct()

scores_df.v2 <- scores_df.v2 %>%
  left_join(meta.v2, by = c("Sample" = "SampleID"))

## Clustering 
clusters.v2 <- cutree(hc.v2, k = 6)

cluster_df.v2 <- data.frame(
  SampleID = names(clusters.v2),
  Cluster = clusters.v2
)

scores_df.v2 <- scores_df.v2 %>%
  left_join(cluster_df.v2, by = c("Sample" = "SampleID"))

## NMDS plot
ggplot(scores_df.v2, aes(NMDS1, NMDS2)) +
  geom_point(aes(color = depth), size = 3) +
  scale_color_viridis_c() +
  theme_minimal() +
  labs(title = "ASV-level NMDS (Depth + Biome)")

## Faceted by biome
ggplot(scores_df.v2, aes(NMDS1, NMDS2)) +
  geom_point(aes(color = depth), size = 2) +
  facet_wrap(~manual.bin) +
  scale_color_viridis_c() +
  theme_minimal()

## Faceted by clusters
ggplot(scores_df.v2, aes(NMDS1, NMDS2)) +
  geom_point(aes(color = depth), size = 2) +
  facet_wrap(~Cluster) +
  scale_color_viridis_c() +
  theme_minimal() +
  labs(title = "ASV-level NMDS split by cluster (k = 6)")

## Ellipses
ggplot(scores_df.v2, aes(NMDS1, NMDS2)) +
       geom_point(aes(color = factor(Cluster)), size = 3) +
       stat_ellipse(aes(group = Cluster)) +
       theme_minimal() +
       labs(title = "NMDS with 95% cluster ellipses")

## Check stress: 0.17
nmds.v2$stress

## PERMANOVA - do groups, either clusters or manual bins, explain centroids in Bray-Curtis space?
adonis2(bray.v2 ~ manual.bin, 
        data = scores_df.v2, 
        method = "bray", 
        permutations = 999) 

## Manual bins (k = 8): R2 = 0.13, p = 0.001

adonis2(bray.v2 ~ Cluster, 
        data = scores_df.v2, 
        method = "bray", 
        permutations = 999)

## Clusters (k = 6): R2 = 0.23, p = 0.001

## ANOSIM
anosim(bray.v2, scores_df.v2$manual.bin) ## R = 0.20, p = 0.001
anosim(bray.v2, scores_df.v2$Cluster) ## R = 0.93, p = 0.001

## Final NMDS plot with biome-depth colors and ANOSIM and PERMANOVA values
bin_cols <- c(
  "Coastal_upper50m"    = "#8c510a",
  "Coastal_50to100m"    = "#dfc27d",
  "Trades_upper50m"     = "#006d2c",
  "Trades_50to100m"     = "#66c2a4",
  "Trades_100to200m"    = "#88419d",
  "Westerlies_upper50m" = "#08519c",
  "Westerlies_50to100m" = "#6baed6",
  "Westerlies_100to200m"= "#f768a1"
)

scores_df.v2$manual.bin <- factor(
  scores_df.v2$manual.bin,
  levels = names(bin_cols)
)

p.nmds <- ggplot(scores_df.v2, aes(NMDS1, NMDS2)) +
  
  geom_point(aes(color = manual.bin, shape = as.factor(Cluster)), size = 2) +
  
  scale_color_manual(values = bin_cols) +
  
  scale_shape_manual(values = c(3, 17, 18, 4, 15, 16), name = "Cluster") +
  
  theme_minimal(base_size = 14) +
  
  labs(
    title = "Global Prochlorococcus Community Structure",
    color = "Biome-depth bins"
  )+
  
  annotate(
    "text",
    x = 2, y = 3.2,
    hjust = 0,
    vjust = 1,
    label = paste(
      "NMDS Stress = 0.17"
    ),
    size = 4
  )+
  
  annotate(
    "text",
    x = 2, y = 3,
    hjust = 0,
    vjust = 1,
    label = paste(
      "Biome-depth (k = 8):",
      "PERMANOVA R² = 0.13; p = 0.001",
      "ANOSIM R = 0.20; p = 0.001",
      "",
      "Clusters (Ward k = 6):",
      "PERMANOVA R² = 0.23; p = 0.001",
      "ANOSIM R = 0.93; p = 0.001",
      sep = "\n"
    ),
    size = 4
  )

ggsave(plot = p.nmds, filename = "__________.png",
       path = "file/path",
       width = 12, height = 8, dpi = 300)
