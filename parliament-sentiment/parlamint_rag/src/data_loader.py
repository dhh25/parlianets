#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module for loading and parsing ParlaMint TEI XML data."""

import os
from pathlib import Path
from lxml import etree
from typing import List, Dict, Any
from tqdm import tqdm

from .config import config

# XML namespaces used in ParlaMint
NS = {
    'tei': 'http://www.tei-c.org/ns/1.0',
    'xml': 'http://www.w3.org/XML/1998/namespace'
}

def parse_tei_file(file_path: Path) -> List[Dict[str, Any]]:
    """Parses a single ParlaMint TEI XML file and extracts utterances with metadata."""
    try:
        tree = etree.parse(str(file_path))
        root = tree.getroot()
    except etree.XMLSyntaxError as e:
        print(f"Error parsing XML file {file_path}: {e}")
        return []

    utterances = []
    doc_id = file_path.stem # Use filename (without .xml) as document ID

    # Find all utterances (u elements)
    for u_element in root.findall('.//tei:u', namespaces=NS):
        speaker_name = u_element.get('who', 'UnknownSpeaker')
        # Refine speaker name if it's a reference to a person element
        if speaker_name.startswith('#'):
            person_id = speaker_name[1:]
            person_element = root.find(f".//tei:person[@xml:id='{person_id}']", namespaces=NS)
            if person_element is not None:
                pers_name_element = person_element.find('.//tei:persName', namespaces=NS)
                if pers_name_element is not None:
                    forename = pers_name_element.findtext('.//tei:forename', namespaces=NS, default='')
                    surname = pers_name_element.findtext('.//tei:surname', namespaces=NS, default='')
                    speaker_name = f"{forename} {surname}".strip()
                    if not speaker_name: # Fallback if forename/surname are empty
                        speaker_name = pers_name_element.text.strip() if pers_name_element.text else person_id

        utterance_id = u_element.get('{http://www.w3.org/XML/1998/namespace}id', None)
        if not utterance_id:
             # Create a simple fallback id if xml:id is missing
            utterance_id = f"{doc_id}_u{len(utterances) + 1}"

        # Extract text from segments (s elements) within the utterance
        segments = u_element.findall('.//tei:seg', namespaces=NS)
        text_parts = []
        for seg in segments:
            seg_text = "".join(seg.itertext()).strip()
            if seg_text:
                text_parts.append(seg_text)
        
        full_text = " ".join(text_parts).strip()
        
        if full_text: # Only add if there is actual text content
            utterances.append({
                'doc_id': doc_id,
                'utterance_id': utterance_id,
                'speaker': speaker_name,
                'text': full_text,
                'source_file': str(file_path.name)
            })

    return utterances

def load_parlamint_data(data_directory: Path = Path(config.paths.raw_data_dir),
                        max_files: int = 0) -> List[Dict[str, Any]]:
    """Loads all ParlaMint TEI XML files from a directory."""
    all_utterances = []
    xml_files = sorted([f for f in data_directory.glob('*.xml')])
    
    if not xml_files:
        print(f"No XML files found in {data_directory}")
        return []

    files_to_process = xml_files
    if max_files > 0:
        files_to_process = xml_files[:max_files]
        print(f"Processing a maximum of {max_files} files.")

    print(f"Found {len(xml_files)} XML files. Processing {len(files_to_process)} files from {data_directory}...")
    for file_path in tqdm(files_to_process, desc="Parsing TEI files"):
        # print(f"Parsing {file_path.name}...")
        utterances = parse_tei_file(file_path)
        all_utterances.extend(utterances)
    
    print(f"Successfully parsed {len(all_utterances)} utterances from {len(files_to_process)} files.")
    return all_utterances

if __name__ == '__main__':
    # Create a dummy TEI XML file for testing
    # Ensure the raw_data_dir exists
    test_raw_data_dir = Path(config.paths.raw_data_dir)
    test_raw_data_dir.mkdir(parents=True, exist_ok=True)

    dummy_xml_content = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0" xml:lang="en">
      <teiHeader>
        <fileDesc>
          <titleStmt>
            <title>Test Parliament Transcript</title>
          </titleStmt>
          <publicationStmt>
            <p>Test data</p>
          </publicationStmt>
          <sourceDesc>
            <p>Manually created for testing.</p>
          </sourceDesc>
        </fileDesc>
        <profileDesc>
            <particDesc>
                <listPerson>
                    <person xml:id="speaker1">
                        <persName>
                            <forename>John</forename>
                            <surname>Doe</surname>
                        </persName>
                    </person>
                     <person xml:id="speaker2">
                        <persName>Jane Smith</persName>
                    </person>
                </listPerson>
            </particDesc>
        </profileDesc>
      </teiHeader>
      <text>
        <body>
          <div type="debateSection">
            <u xml:id="u1" who="#speaker1">
              <seg xml:id="u1.s1">This is the first sentence.</seg>
              <seg xml:id="u1.s2">And this is the second one.</seg>
            </u>
            <u xml:id="u2" who="#speaker2">
              <seg xml:id="u2.s1">Jane Smith is speaking now.</seg>
              <seg xml:id="u2.s2">She will continue for a moment.</seg>
              <seg xml:id="u2.s3">And will end here.</seg>
            </u>
            <u xml:id="u3" who="#unknownSpeaker">
                <seg xml:id="u3.s1">Unknown speaker.</seg>
            </u>
          </div>
        </body>
      </text>
    </TEI>
    """
    dummy_file_path = test_raw_data_dir / "dummy_parlamint_en_01.xml" # Changed filename
    with open(dummy_file_path, 'w', encoding='utf-8') as f:
        f.write(dummy_xml_content)
    
    print(f"Created dummy TEI file: {dummy_file_path}")

    # Test the loader with the dummy file
    print(f"\n--- Testing data_loader with max_files=1 ---")
    loaded_data_test_max = load_parlamint_data(max_files=1)
    for utterance in loaded_data_test_max:
        print(utterance)
    
    # Create another dummy file to test loading multiple files
    dummy_xml_content_2 = dummy_xml_content.replace("<title>Test Parliament Transcript</title>", "<title>Test Parliament Transcript 2</title>")
    dummy_xml_content_2 = dummy_xml_content_2.replace("This is the first sentence.", "A completely different opening.")
    dummy_file_path_2 = test_raw_data_dir / "dummy_parlamint_en_02.xml" # Changed filename
    with open(dummy_file_path_2, 'w', encoding='utf-8') as f:
        f.write(dummy_xml_content_2)
    print(f"Created second dummy TEI file: {dummy_file_path_2}")

    print(f"\n--- Testing data_loader with all files in raw directory ---")
    loaded_data_all = load_parlamint_data()
    print(f"Total utterances loaded: {len(loaded_data_all)}")
    # for utterance in loaded_data_all:
    #     print(utterance)

    # Clean up dummy files - note: in testing these are often left for inspection
    # For automated test runs, you might want to uncomment cleanup.
    # os.remove(dummy_file_path)
    # os.remove(dummy_file_path_2)
    # print("Cleaned up dummy files.")
    print("\nNOTE: Dummy files are not automatically cleaned up to allow inspection.")
    print(f"Please remove them manually from: {test_raw_data_dir} if they are from a previous Finnish run.") 