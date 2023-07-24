# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 09:45:39 2023

@author: YOGB
"""
import streamlit as st
import base64
from streamlit_folium import st_folium, folium_static
from module.wibopile_streamlit import PileCalculator, PileCatalog, PileChart, PileSoilProfile, PileSummary, PileReport

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_state(i):
    st.session_state.stage = i

def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1100" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

@st.cache_data
def generate_report(
        filename,
        soil_table,
        pile_catalog_table,
        input_parameter_table,
        selected_pile,
        summary_capacity_table,
        summary_kv_table,
        kh_tables,
        liquefaction_checkbox,
        summary_capacity_liq):
    report = PileReport()
    report.generate_report(filename,
                           soil_table,
                           pile_catalog_table,
                           input_parameter_table,
                           selected_pile,
                           summary_capacity_table,
                           summary_kv_table,
                           kh_tables,
                           liquefaction_checkbox,
                           summary_capacity_liq)

@st.cache_data
def main_calculation(selected_pile,
                     borehole_assigned,
                     index_pile,
                     catalog_prop,
                     SF_compression,
                     SF_tension,
                     cohesiveapproach,
                     granularapproachshaft,
                     granularapproachendB,
                     liquefaction_checkbox,
                     PILE_DEPTH,
                     COMP_LOAD):
    all_result_table, all_kv_table, all_liquefaction_table, all_summary, all_summary_liq = [],[],[],[],[]
    all_kh_dict = {}
    for i in range(len(selected_pile)):
        pilemodel = PileCalculator(borehole=borehole_assigned)
        index = index_pile[i]
        output_table = pilemodel.calculate_bearing_capacity(
            pile_parameter_table = catalog_prop.iloc[[index]],
            SF_compression = SF_compression,
            SF_tension = SF_tension,
            cohesive_approach = cohesiveapproach,
            granular_approach_shaft = granularapproachshaft,
            granular_approach_endB = granularapproachendB,
            )
        
        if liquefaction_checkbox:
            output_table_liquefied = pilemodel.calculate_bearing_capacity_liquefied(
                calculation = output_table,
                reference = edited_liquefaction_table,
                pile_parameter_table = catalog_prop.iloc[[index]],
                SF_compression = SF_compression,
                SF_tension = SF_tension)
            
            summary_liq = pilemodel.get_bearing_capacity_from_depth(data_table = output_table_liquefied, depth = PILE_DEPTH)
        
        kv_table = pilemodel.calculate_kv(
            data_table=output_table,
            pile_depth=PILE_DEPTH,
            load_compression=COMP_LOAD,
            load_tension=TENS_LOAD,
            area_section=catalog_prop['Area_middle'].iloc[index]
            )
        
        kh_table = pilemodel.calculate_kh(data_table = output_table,
                                          pile_diameter = catalog_prop['Diameter (m)'].iloc[index])
        
        summary = pilemodel.get_bearing_capacity_from_depth(data_table = output_table, depth = PILE_DEPTH)
        
        
        all_result_table.append(output_table)
        if liquefaction_checkbox:
            all_liquefaction_table.append(output_table_liquefied)
            all_summary_liq.append(summary_liq)
        all_kv_table.append(kv_table[1])
        all_summary.append(summary)
        all_kh_dict[selected_pile[i]] = kh_table[1]
    
    return all_result_table, all_kv_table, all_liquefaction_table, all_summary, all_kh_dict, all_summary_liq


st.set_page_config(page_title="Pile Capacity", layout="wide")
st.header("Pile Bearing Capacity with Varied Pile Properties")

with st.sidebar:
    uploaded_files = st.file_uploader("Upload XLSX file(s):", type=["xlsx"], accept_multiple_files=True)
    
    if uploaded_files:
        listfile = [i.name for i in uploaded_files]
        select = st.sidebar.selectbox("Select soil profile:", options=listfile)
        index_BH = listfile.index(select)
    
    SF_compression = st.number_input("SF compression:", value=2.5)
    SF_tension = st.number_input("SF tension:", value=3.0)
    option_cohesive_approach = ["API", "NAVFAC"]
    cohesiveapproach = st.selectbox("Cohesive approach", option_cohesive_approach)
    option_granular_approach_shaft = ["API", "NAVFAC", "OCDI(NSPT)"]
    granularapproachshaft = st.selectbox("Granular approach shaft", option_granular_approach_shaft)
    option_granular_approach_endB = ["NAVFAC", "API"]
    granularapproachendB = st.selectbox("Granular approach endB", option_granular_approach_endB)
    PILE_DEPTH = st.number_input('Pile depth (m):')
    COMP_LOAD = st.number_input('Compression load (kN):')
    TENS_LOAD = st.number_input('Tension load (kN):')
    liquefaction_checkbox = st.checkbox("INCLUDE LIQUEFACTION")
    if liquefaction_checkbox:
        COMP_LOAD_DYN = st.number_input('Compression load - Dynamic Case (kN):')
        TENS_LOAD_DYN = st.number_input('Tension load - Dynamic Case (kN):')

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Pile Catalog", "Borehole", "Charts", "Summary and Tables", "Report", "References"])

with tab1:

    catalog = PileCatalog()
    catalog_table = catalog.get_catalog_table()

    st.subheader("Pile catalog table")
    catalog_table_dyn = st.data_editor(catalog_table,
                   column_config = {
                       "Type": st.column_config.SelectboxColumn(
                           "Type",
                           width = "small",
                           options = [
                               "circular",
                               "square",
                               "circ_hollow",
                               ]
                           ),
                       "Material": st.column_config.SelectboxColumn(
                           "Material",
                           width = "small",
                           options = [
                               "concrete",
                               "steel",
                               ]
                           ),
                       "End bearing": st.column_config.SelectboxColumn(
                           "End bearing",
                           width = "small",
                           options = [
                               "with",
                               "without",
                               ]
                           ),
                       },
                   hide_index = True,
                   use_container_width=True,
                   num_rows='dynamic',
                   )
    
    catalog_prop = catalog.get_catalog_properties(catalog_table_dyn)
    
    pile_names = catalog_table_dyn["Name"].tolist()

if uploaded_files and select: 
    with tab2:
        col2a, col2b = st.columns([2.5,3])
        
        with col2a:
            borelog = PileSoilProfile(uploaded_files[index_BH])
            borelog.get_soil_profile_chart_mpl()
            soil_table = borelog.get_soil_profile_table()
            
            tab2a, tab2b, tab2c = st.tabs(["Soil Table", "Soil Profile Chart", "Borehole Information"])
            with tab2a:
                edited_table = st.data_editor(soil_table,
                                              column_config = {
                                                  "soil": st.column_config.SelectboxColumn(
                                                      "soil",
                                                      width = "small",
                                                      options = [
                                                          "clay",
                                                          "silt",
                                                          "sand loose",
                                                          "sand medium",
                                                          "sand dense",
                                                          "sand very dense",
                                                          ]
                                                      ),
                                                  "state": st.column_config.SelectboxColumn(
                                                      "state",
                                                      width = "small",
                                                      options = [
                                                          "UC",
                                                          "NC",
                                                          "OC",
                                                          ]
                                                      )
                                                  },
                                              hide_index = True,
                                              use_container_width = True,
                                              )
                
                borehole_assigned = borelog.assign_borehole(edited_table)
                
                
                if liquefaction_checkbox:
                
                    liquefaction_table = borelog.assign_liquefaction(edited_table)
                    edited_liquefaction_table = st.data_editor(liquefaction_table,
                                                  column_config = {
                                                      "liquefied?": st.column_config.SelectboxColumn(
                                                          "liquefied?",
                                                          width = "medium",
                                                          options = [
                                                              "yes",
                                                              "no",
                                                              ]
                                                          )                                                 
                                                      },
                                                  hide_index = True,
                                                  use_container_width = True,
                                                  )
                
                
            with tab2b: #chart profile borelog
                borelog_chart = borelog.get_soil_profile_chart(edited_table)
                st.plotly_chart(borelog_chart)
            
            with tab2c:
                bore_general_table = borelog.get_general_table()
                st.dataframe(bore_general_table, use_container_width=True)
            
        with col2b:
            m = borelog.get_map()
            mymap = st_folium(m, width=800, returned_objects=[])
    
    
    with tab3:
        
        col3a, col3b = st.columns([2.5,3])
        
        with col3a:
            
            selected_pile = st.multiselect("Select pile name from the pile catalog:", options = pile_names)
            catalog_selected = catalog_table_dyn[catalog_table_dyn['Name'].isin(selected_pile)]
            
            index_pile = []
            for i in range(len(selected_pile)):
                index_pile.append(pile_names.index(selected_pile[i]))
            
            field_list = ["compression_ultimate", "compression_allowed", "tension_ultimate", "tension_allowed"]
            selected_field = st.selectbox("Select field to be displayed", options=field_list, index=1)
            
            if st.session_state.stage == 0:
                st.button('CALCULATE', on_click=set_state, args=[1], use_container_width=True)
                st.warning('To start calculation, click "CALCULATE" button and go to the next process. To perform another calculation with different input parameters, click "START OVER" button before changing the input parameters.',
                           icon="⚠️")
                # st.button('START OVER', on_click=set_state, args=[0], use_container_width=True)
            
            # button_calc = st.button('CALCULATE', use_container_width=True)
            if st.session_state.stage >= 1:
                st.button('START OVER', on_click=set_state, args=[0], use_container_width=True)
                st.warning('To start calculation, click "CALCULATE" button and go to the next process. To perform another calculation with different input parameters, click "START OVER" button before changing the input parameters.',
                           icon="⚠️")
                

                all_result_table, all_kv_table, all_liquefaction_table, all_summary, all_kh_dict, all_summary_liq = main_calculation(selected_pile,
                                                                                                                    borehole_assigned,
                                                                                                                    index_pile,
                                                                                                                    catalog_prop,
                                                                                                                    SF_compression,
                                                                                                                    SF_tension,
                                                                                                                    cohesiveapproach,
                                                                                                                    granularapproachshaft,
                                                                                                                    granularapproachendB,
                                                                                                                    liquefaction_checkbox,
                                                                                                                    PILE_DEPTH,
                                                                                                                    COMP_LOAD)
            
            
                with col3b:
                    
                    tab3a,tab3b = st.tabs(["non-liquefied","liquefied"])
                    chart = PileChart()
                    chart.get_multi_bearing_capacity_chart_mpl(all_result_table, selected_pile, 'compression_allowed', "chart_capacity_multi_mpl_CA.png")
                    chart.get_multi_bearing_capacity_chart_mpl(all_result_table, selected_pile, 'tension_allowed', "chart_capacity_multi_mpl_TA.png")
                    
                    FIG1 = chart.get_multi_bearing_capacity_chart(all_result_table, selected_pile, field = selected_field)
                    
                    if liquefaction_checkbox:
                        FIG2 = chart.get_multi_bearing_capacity_chart(all_liquefaction_table, selected_pile, field = selected_field)
                        
                        chart.get_multi_bearing_capacity_chart_mpl(all_liquefaction_table, selected_pile, 'compression_allowed', "chart_capacity_multi_mpl_liq_CA.png")
                        chart.get_multi_bearing_capacity_chart_mpl(all_liquefaction_table, selected_pile, 'tension_allowed', "chart_capacity_multi_mpl_liq_TA.png")
                        
                    with tab3a:
                        st.plotly_chart(FIG1, use_container_width = True)
                    
                    with tab3b:
                        if liquefaction_checkbox:
                            st.plotly_chart(FIG2, use_container_width = True)
    
                with tab4:
                    
                    if selected_pile:
                        pile_kh = st.selectbox("Select pile:", options=selected_pile)
                        pile_kh_index = selected_pile.index(pile_kh)
                        
                        tab3a, tab3b = st.tabs(["Summary", "Calculation Table"])
                        
                        with tab3a:
            
                            overview = PileSummary()
                            summary_capacity, summary_kv = overview.get_summary_multiple_data(
                                result_capacity_table = all_summary,
                                result_kv_table = all_kv_table,
                                key_name = selected_pile,
                                load_compression = COMP_LOAD,
                                load_tension = TENS_LOAD,
                                )
                            
                            if liquefaction_checkbox:
                                summary_capacity_liq, summary_kv_liq = overview.get_summary_multiple_data(
                                    result_capacity_table = all_summary_liq,
                                    result_kv_table = all_kv_table,
                                    key_name = selected_pile,
                                    load_compression = COMP_LOAD_DYN,
                                    load_tension = TENS_LOAD_DYN,
                                    )
                            
                            input_summary = overview.get_input_summary(
                                SF_compression,
                                SF_tension,
                                cohesiveapproach,
                                granularapproachshaft,
                                granularapproachendB,
                                PILE_DEPTH,
                                COMP_LOAD,
                                TENS_LOAD
                                )
                            
                            st.subheader("Summary of bearing capacity and vertical spring stiffness coefficient")
                            st.markdown("STATIC CASE (NON-LIQUEFIED)")
                            st.dataframe(summary_capacity, use_container_width=True)
                            st.dataframe(summary_kv, use_container_width=True)
                            
                            if liquefaction_checkbox:
                                st.markdown("DYNAMIC CASE (LIQUEFIED)")
                                st.dataframe(summary_capacity_liq, use_container_width=True)
                            
                            st.subheader("Summary of horizontal spring stiffness coefficient")
                            st.markdown("STATIC CASE (NON-LIQUEFIED) ONLY")
                            kh_table = all_kh_dict[pile_kh]
                            st.dataframe(kh_table, use_container_width=True)
                        
                        with tab3b:
                            st.markdown("STATIC CASE (NON-LIQUEFIED)")
                            st.dataframe(all_result_table[pile_kh_index], use_container_width=True)
                            
                            if liquefaction_checkbox:
                                st.markdown("DYNAMIC CASE (LIQUEFIED)")
                                st.dataframe(all_liquefaction_table[pile_kh_index], use_container_width=True)
                            
                    else:
                        st.warning('First select pile name from the catalog!')
            
                
                with tab5:
                    st.button('Generate Report', on_click=set_state, args=[2])
                    
                    # report_button = st.button('Generate Report')
                    if st.session_state.stage >= 2:
                        if liquefaction_checkbox:
                            generate_report(
                                    filename = "report_pile.pdf",
                                    soil_table = edited_table,
                                    pile_catalog_table = catalog_table_dyn,
                                    input_parameter_table = input_summary,
                                    selected_pile = catalog_selected,
                                    summary_capacity_table = summary_capacity,
                                    summary_kv_table = summary_kv,
                                    kh_tables = all_kh_dict,
                                    liquefaction_checkbox = True,
                                    summary_capacity_liq = summary_capacity_liq
                                    )
                        
                        else:
                            generate_report(
                                    filename = "report_pile.pdf",
                                    soil_table = edited_table,
                                    pile_catalog_table = catalog_table_dyn,
                                    input_parameter_table = input_summary,
                                    selected_pile = catalog_selected,
                                    summary_capacity_table = summary_capacity,
                                    summary_kv_table = summary_kv,
                                    kh_tables = all_kh_dict,
                                    liquefaction_checkbox = False,
                                    summary_capacity_liq = None
                                    )
                        
                        with open("report_pile.pdf", "rb") as pdf_file:
                            PDFbyte = pdf_file.read()
                
                        st.download_button(label="Download Report", 
                                data=PDFbyte,
                                file_name="report_pile.pdf",
                                mime='application/octet-stream')

with tab6:
    show_pdf("memorandum_pile review approach_v4.pdf")

            
        
