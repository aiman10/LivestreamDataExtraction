"""
influx_importer.py
One-shot script to import CSV data into InfluxDB for Grafana visualization.

Usage:
    1. Start InfluxDB + Grafana: docker-compose up -d
    2. Run this script: python influx_importer.py

Prerequisites:
    pip install influxdb-client
"""

import csv
import sys
from datetime import datetime

import config

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
except ImportError:
    print("influxdb-client not installed. Run: pip install influxdb-client")
    sys.exit(1)


def import_metrics(write_api, bucket):
    """Import metrics.csv into InfluxDB."""
    path = config.METRICS_CSV_PATH
    count = 0

    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                point = Point("scene_metrics")

                # Timestamp
                point.time(row["timestamp"], WritePrecision.S)

                # Numeric fields (stored as fields in InfluxDB)
                numeric_fields = {
                    "person_count": int,
                    "vehicle_count": int,
                    "bicycle_count": int,
                    "umbrella_count": int,
                    "backpack_count": int,
                    "suitcase_count": int,
                    "dog_count": int,
                    "moving_object_count": int,
                    "motion_pct": float,
                    "pedestrian_vehicle_ratio": float,
                    "dominant_flow_dir": float,
                    "avg_flow_speed": float,
                    "brightness": float,
                    "contrast": float,
                    "saturation": float,
                    "color_temp": float,
                    "green_ratio": float,
                    "wave_count": int,
                    "total_waves": int,
                    "photo_stop_count": int,
                    "total_photo_stops": int,
                    "friendliness_index": float,
                }

                for field_name, cast_fn in numeric_fields.items():
                    value = row.get(field_name, "")
                    if value != "":
                        try:
                            point.field(field_name, cast_fn(value))
                        except (ValueError, TypeError):
                            pass

                # Boolean fields
                for bool_field in ["is_raining", "is_daytime"]:
                    value = row.get(bool_field, "")
                    if value != "":
                        point.field(bool_field, value.lower() in ("true", "1", "yes"))

                # Tag fields (indexed, used for filtering/grouping)
                for tag_field in ["crowd_density", "activity_level", "scene_state", "friendliness_level"]:
                    value = row.get(tag_field, "")
                    if value:
                        point.tag(tag_field, value)

                write_api.write(bucket=bucket, record=point)
                count += 1

                if count % 100 == 0:
                    print(f"  Imported {count} metric rows...")

    except FileNotFoundError:
        print(f"  File not found: {path}")
        return 0

    return count


def import_events(write_api, bucket):
    """Import events.csv into InfluxDB."""
    path = config.EVENTS_CSV_PATH
    count = 0

    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                point = Point("scene_events")
                point.time(row["timestamp"], WritePrecision.S)

                # Tags for filtering
                point.tag("event_type", row.get("event_type", ""))
                point.tag("severity", row.get("severity", ""))
                point.tag("scene_state", row.get("scene_state", ""))

                # Fields
                metric_val = row.get("metric_value", "")
                if metric_val:
                    try:
                        point.field("metric_value", float(metric_val))
                    except ValueError:
                        pass

                threshold_val = row.get("threshold", "")
                if threshold_val:
                    try:
                        point.field("threshold", float(threshold_val))
                    except ValueError:
                        pass

                point.field("description", row.get("description", ""))

                write_api.write(bucket=bucket, record=point)
                count += 1

    except FileNotFoundError:
        print(f"  File not found: {path}")
        return 0

    return count


def main():
    print("=" * 60)
    print("InfluxDB CSV Importer")
    print("=" * 60)
    print(f"  InfluxDB URL: {config.INFLUXDB_URL}")
    print(f"  Organization: {config.INFLUXDB_ORG}")
    print(f"  Bucket:       {config.INFLUXDB_BUCKET}")
    print()

    # Connect to InfluxDB
    client = InfluxDBClient(
        url=config.INFLUXDB_URL,
        token=config.INFLUXDB_TOKEN,
        org=config.INFLUXDB_ORG,
    )

    # Test connection
    try:
        health = client.health()
        if health.status != "pass":
            print(f"InfluxDB health check failed: {health.message}")
            sys.exit(1)
        print(f"  Connected to InfluxDB (version {health.version})")
    except Exception as e:
        print(f"Cannot connect to InfluxDB: {e}")
        print("Make sure InfluxDB is running: docker-compose up -d")
        sys.exit(1)

    write_api = client.write_api(write_options=SYNCHRONOUS)

    # Import metrics
    print("\nImporting metrics...")
    metrics_count = import_metrics(write_api, config.INFLUXDB_BUCKET)
    print(f"  Done: {metrics_count} metric rows imported")

    # Import events
    print("\nImporting events...")
    events_count = import_events(write_api, config.INFLUXDB_BUCKET)
    print(f"  Done: {events_count} event rows imported")

    # Summary
    print("\n" + "=" * 60)
    print(f"Total imported: {metrics_count} metrics + {events_count} events")
    print()
    print("Next steps:")
    print("  1. Open Grafana at http://localhost:3000 (admin/admin)")
    print("  2. Add InfluxDB data source:")
    print(f"     - URL: {config.INFLUXDB_URL}")
    print(f"     - Organization: {config.INFLUXDB_ORG}")
    print(f"     - Token: {config.INFLUXDB_TOKEN}")
    print(f"     - Default bucket: {config.INFLUXDB_BUCKET}")
    print("  3. Import the dashboard from grafana/dashboard.json")
    print("=" * 60)

    client.close()


if __name__ == "__main__":
    main()
