# Specification SP-DRN-014: Battery pack for the IV-3 inspection drone

Context. The IV-3 is a quadcopter used by our field teams to inspect wind-turbine blades and overhead power lines. The current battery supplier is discontinuing its product line, so a replacement pack must be selected from the shortlisted catalogue. This document lists what the replacement must, or preferably should, achieve. Requirements stated with "shall" or "must", or whose violation makes the pack unusable, are mandatory. Recommendations ("should", "preferably", "desirable", "would be appreciated") do not by themselves disqualify a candidate.

[R01] A typical inspection mission consists of 30 minutes of hovering and slow translation, during which the propulsion system and payload draw 180 W on average. The pack shall store enough energy to complete such a mission without landing. Reserve margins are handled by the flight controller and are not part of this requirement.

[R02] With the payload installed, the airframe is 800 g below its maximum take-off mass. Whatever pack is chosen must not consume more than this margin.

[R03] Missions are flown all year round. The pack shall deliver its rated performance at ambient temperatures from −10 °C (winter inspections) up to 45 °C (summer, in direct sun on the tarmac).

[R04] Packs are shipped to field teams by air freight. A UN38.3 test summary is a precondition for the acceptance of any lithium battery by our logistics department.

[R05] The electronic speed controllers and the power distribution board are rated for six lithium cells in series. Any other series configuration cannot be used.

[R06] During climb and gust rejection the motors draw up to 40 A in total, sustained for more than a minute. The pack shall support this current continuously.

[R07] To keep the cost of ownership acceptable, the pack should still provide 80 % of its initial capacity after at least 300 full charge–discharge cycles.

[R08] The existing harness ends in an XT90 plug. Delivery with a matching connector would avoid re-soldering and is therefore preferred.

[R09] Field teams carry a single charger. It is desirable that a pack can be fully recharged within 90 minutes, so that two packs can be rotated during a working day.

[R10] The battery bay accepts packs up to 160 mm long; width and height are not limiting for any 6S pack in the catalogue. Longer packs cannot be mounted.

[R11] A smart battery-management system that reports the state of charge to the flight controller would be appreciated.

[R12] Inspections are occasionally interrupted by light rain. Preferably, the pack enclosure should be rated at least IP54.

Selection. Among the candidates that satisfy every mandatory requirement, the purchasing department selects the one with the lowest unit price; if two candidates have the same price, the lighter one is chosen.

Out of scope. Charger compatibility, storage conditions and end-of-life recycling are covered by separate documents.
