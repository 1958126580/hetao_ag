"""
Feed Inventory Module

Feed inventory management and cost analysis:
- Stock tracking
- Purchase and usage records
- Cost analysis
- Forecasting

Example:
    >>> inventory = FeedInventory()
    >>> inventory.add_purchase("corn_grain", 1000, price=250)
    >>> inventory.record_usage("corn_grain", 50, "herd_a")
    >>> report = inventory.get_inventory_report()
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeedPurchase:
    """
    Feed purchase record.

    Attributes:
        id: Purchase ID
        feed_name: Feed type
        quantity: Quantity purchased (kg)
        unit_price: Price per kg
        total_cost: Total purchase cost
        supplier: Supplier name
        date: Purchase date
        batch_number: Batch/lot number
        quality_grade: Quality grade
        expiry_date: Expiration date
    """
    id: str
    feed_name: str
    quantity: float
    unit_price: float
    total_cost: float
    supplier: str = ""
    date: date = field(default_factory=date.today)
    batch_number: str = ""
    quality_grade: str = ""
    expiry_date: Optional[date] = None


@dataclass
class FeedUsage:
    """
    Feed usage record.

    Attributes:
        id: Usage ID
        feed_name: Feed type
        quantity: Quantity used (kg)
        destination: Usage destination (herd, pen, etc.)
        date: Usage date
        recorded_by: Person recording
        notes: Additional notes
    """
    id: str
    feed_name: str
    quantity: float
    destination: str
    date: date = field(default_factory=date.today)
    recorded_by: str = ""
    notes: str = ""


@dataclass
class FeedStock:
    """
    Current feed stock status.

    Attributes:
        feed_name: Feed type
        quantity: Current quantity (kg)
        avg_cost: Average cost per kg
        last_purchase: Last purchase date
        last_usage: Last usage date
        batches: List of batch details
    """
    feed_name: str
    quantity: float
    avg_cost: float
    last_purchase: Optional[date] = None
    last_usage: Optional[date] = None
    batches: List[Dict[str, Any]] = field(default_factory=list)


class FeedInventory:
    """
    Feed inventory management system.

    Tracks feed purchases, usage, and stock levels
    with FIFO cost accounting.

    Example:
        >>> inventory = FeedInventory()
        >>> inventory.add_purchase("corn_grain", 1000, 0.25)
        >>> inventory.record_usage("corn_grain", 50)
        >>> stock = inventory.get_stock("corn_grain")
    """

    def __init__(self):
        """Initialize feed inventory."""
        self._purchases: List[FeedPurchase] = []
        self._usage: List[FeedUsage] = []
        self._stock: Dict[str, List[Dict[str, Any]]] = {}  # FIFO batches
        self._reorder_levels: Dict[str, float] = {}
        self._purchase_counter = 0
        self._usage_counter = 0

    def set_reorder_level(
        self,
        feed_name: str,
        level: float,
    ) -> None:
        """Set reorder alert level for feed."""
        self._reorder_levels[feed_name] = level

    def add_purchase(
        self,
        feed_name: str,
        quantity: float,
        unit_price: float,
        supplier: str = "",
        purchase_date: Optional[date] = None,
        batch_number: str = "",
        quality_grade: str = "",
        expiry_date: Optional[date] = None,
    ) -> FeedPurchase:
        """
        Record feed purchase.

        Args:
            feed_name: Feed type name
            quantity: Quantity purchased (kg)
            unit_price: Price per kg
            supplier: Supplier name
            purchase_date: Date of purchase
            batch_number: Batch/lot number
            quality_grade: Quality grade
            expiry_date: Expiration date

        Returns:
            FeedPurchase record
        """
        if purchase_date is None:
            purchase_date = date.today()

        self._purchase_counter += 1
        purchase_id = f"PUR-{self._purchase_counter:06d}"

        purchase = FeedPurchase(
            id=purchase_id,
            feed_name=feed_name,
            quantity=quantity,
            unit_price=unit_price,
            total_cost=quantity * unit_price,
            supplier=supplier,
            date=purchase_date,
            batch_number=batch_number,
            quality_grade=quality_grade,
            expiry_date=expiry_date,
        )

        self._purchases.append(purchase)

        # Add to stock (FIFO)
        if feed_name not in self._stock:
            self._stock[feed_name] = []

        self._stock[feed_name].append({
            "batch_number": batch_number,
            "quantity": quantity,
            "unit_price": unit_price,
            "purchase_date": purchase_date,
            "expiry_date": expiry_date,
        })

        logger.info(f"Purchase recorded: {quantity} kg of {feed_name}")

        return purchase

    def record_usage(
        self,
        feed_name: str,
        quantity: float,
        destination: str = "",
        usage_date: Optional[date] = None,
        recorded_by: str = "",
        notes: str = "",
    ) -> Tuple[FeedUsage, float]:
        """
        Record feed usage.

        Args:
            feed_name: Feed type name
            quantity: Quantity used (kg)
            destination: Usage destination
            usage_date: Date of usage
            recorded_by: Person recording
            notes: Additional notes

        Returns:
            Tuple of (FeedUsage record, actual cost)
        """
        if usage_date is None:
            usage_date = date.today()

        # Check stock availability
        available = self.get_stock_level(feed_name)
        if quantity > available:
            logger.warning(
                f"Insufficient stock: requested {quantity}, available {available}"
            )
            quantity = available

        self._usage_counter += 1
        usage_id = f"USE-{self._usage_counter:06d}"

        usage = FeedUsage(
            id=usage_id,
            feed_name=feed_name,
            quantity=quantity,
            destination=destination,
            date=usage_date,
            recorded_by=recorded_by,
            notes=notes,
        )

        self._usage.append(usage)

        # Deduct from stock (FIFO)
        actual_cost = self._deduct_stock(feed_name, quantity)

        logger.info(f"Usage recorded: {quantity} kg of {feed_name}")

        return usage, actual_cost

    def _deduct_stock(
        self,
        feed_name: str,
        quantity: float,
    ) -> float:
        """Deduct quantity from stock using FIFO."""
        if feed_name not in self._stock:
            return 0.0

        batches = self._stock[feed_name]
        remaining = quantity
        total_cost = 0.0

        while remaining > 0 and batches:
            batch = batches[0]
            if batch["quantity"] <= remaining:
                # Use entire batch
                total_cost += batch["quantity"] * batch["unit_price"]
                remaining -= batch["quantity"]
                batches.pop(0)
            else:
                # Partial batch
                total_cost += remaining * batch["unit_price"]
                batch["quantity"] -= remaining
                remaining = 0

        return total_cost

    def get_stock_level(self, feed_name: str) -> float:
        """Get current stock level for feed."""
        if feed_name not in self._stock:
            return 0.0
        return sum(b["quantity"] for b in self._stock[feed_name])

    def get_stock(self, feed_name: str) -> FeedStock:
        """Get detailed stock information."""
        batches = self._stock.get(feed_name, [])
        quantity = sum(b["quantity"] for b in batches)

        if quantity > 0:
            avg_cost = sum(b["quantity"] * b["unit_price"] for b in batches) / quantity
        else:
            avg_cost = 0.0

        # Find last purchase and usage
        purchases = [p for p in self._purchases if p.feed_name == feed_name]
        usages = [u for u in self._usage if u.feed_name == feed_name]

        return FeedStock(
            feed_name=feed_name,
            quantity=quantity,
            avg_cost=avg_cost,
            last_purchase=purchases[-1].date if purchases else None,
            last_usage=usages[-1].date if usages else None,
            batches=batches,
        )

    def get_expiring_stock(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get stock expiring within period."""
        cutoff = date.today() + timedelta(days=days)
        expiring = []

        for feed_name, batches in self._stock.items():
            for batch in batches:
                if batch["expiry_date"] and batch["expiry_date"] <= cutoff:
                    days_until = (batch["expiry_date"] - date.today()).days
                    expiring.append({
                        "feed_name": feed_name,
                        "batch_number": batch["batch_number"],
                        "quantity": batch["quantity"],
                        "expiry_date": batch["expiry_date"],
                        "days_until_expiry": days_until,
                        "expired": days_until < 0,
                    })

        return sorted(expiring, key=lambda x: x["expiry_date"])

    def get_low_stock_alerts(self) -> List[Dict[str, Any]]:
        """Get feeds below reorder level."""
        alerts = []

        for feed_name, level in self._reorder_levels.items():
            current = self.get_stock_level(feed_name)
            if current < level:
                alerts.append({
                    "feed_name": feed_name,
                    "current_stock": current,
                    "reorder_level": level,
                    "shortage": level - current,
                })

        return alerts

    def get_inventory_report(self) -> Dict[str, Any]:
        """Generate comprehensive inventory report."""
        all_feeds = set(self._stock.keys())
        for p in self._purchases:
            all_feeds.add(p.feed_name)

        stocks = {}
        total_value = 0.0

        for feed_name in all_feeds:
            stock = self.get_stock(feed_name)
            value = stock.quantity * stock.avg_cost
            stocks[feed_name] = {
                "quantity": stock.quantity,
                "avg_cost": stock.avg_cost,
                "value": value,
            }
            total_value += value

        return {
            "date": date.today(),
            "stocks": stocks,
            "total_items": len(stocks),
            "total_value": total_value,
            "low_stock_count": len(self.get_low_stock_alerts()),
            "expiring_count": len(self.get_expiring_stock(30)),
        }


class CostAnalyzer:
    """
    Feed cost analysis and forecasting.

    Analyzes feed costs, usage patterns, and
    provides budget forecasting.

    Example:
        >>> analyzer = CostAnalyzer(inventory)
        >>> report = analyzer.analyze_costs(start_date, end_date)
        >>> forecast = analyzer.forecast_costs(days=30)
    """

    def __init__(self, inventory: FeedInventory):
        """
        Initialize cost analyzer.

        Args:
            inventory: FeedInventory instance
        """
        self.inventory = inventory

    def analyze_costs(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Analyze feed costs for period.

        Args:
            start_date: Analysis start date
            end_date: Analysis end date

        Returns:
            Dict with cost analysis
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        # Filter purchases
        purchases = [
            p for p in self.inventory._purchases
            if start_date <= p.date <= end_date
        ]

        # Filter usage
        usages = [
            u for u in self.inventory._usage
            if start_date <= u.date <= end_date
        ]

        # Aggregate by feed
        by_feed = {}
        for purchase in purchases:
            if purchase.feed_name not in by_feed:
                by_feed[purchase.feed_name] = {
                    "purchased_qty": 0,
                    "purchased_cost": 0,
                    "used_qty": 0,
                }
            by_feed[purchase.feed_name]["purchased_qty"] += purchase.quantity
            by_feed[purchase.feed_name]["purchased_cost"] += purchase.total_cost

        for usage in usages:
            if usage.feed_name not in by_feed:
                by_feed[usage.feed_name] = {
                    "purchased_qty": 0,
                    "purchased_cost": 0,
                    "used_qty": 0,
                }
            by_feed[usage.feed_name]["used_qty"] += usage.quantity

        # Calculate totals
        total_purchased = sum(p.total_cost for p in purchases)
        total_qty_purchased = sum(p.quantity for p in purchases)
        total_qty_used = sum(u.quantity for u in usages)

        days = (end_date - start_date).days + 1

        return {
            "period": {
                "start": start_date,
                "end": end_date,
                "days": days,
            },
            "summary": {
                "total_purchased_cost": total_purchased,
                "total_purchased_qty": total_qty_purchased,
                "total_used_qty": total_qty_used,
                "avg_daily_cost": total_purchased / days if days > 0 else 0,
                "avg_daily_usage": total_qty_used / days if days > 0 else 0,
            },
            "by_feed": by_feed,
            "n_purchases": len(purchases),
            "n_usage_records": len(usages),
        }

    def forecast_costs(
        self,
        days: int = 30,
        historical_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Forecast feed costs.

        Args:
            days: Days to forecast
            historical_days: Historical days to base forecast on

        Returns:
            Dict with forecast
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=historical_days)

        analysis = self.analyze_costs(start_date, end_date)

        daily_cost = analysis["summary"]["avg_daily_cost"]
        daily_usage = analysis["summary"]["avg_daily_usage"]

        # Forecast by feed
        feed_forecasts = {}
        for feed_name, data in analysis["by_feed"].items():
            daily_qty = data["used_qty"] / historical_days
            stock = self.inventory.get_stock(feed_name)

            days_of_stock = stock.quantity / daily_qty if daily_qty > 0 else float('inf')

            feed_forecasts[feed_name] = {
                "daily_usage": daily_qty,
                "forecasted_usage": daily_qty * days,
                "current_stock": stock.quantity,
                "days_of_stock": days_of_stock,
                "will_run_out": days_of_stock < days,
                "reorder_needed": stock.quantity - daily_qty * days,
            }

        return {
            "forecast_period_days": days,
            "based_on_days": historical_days,
            "forecasted_total_cost": daily_cost * days,
            "forecasted_total_usage": daily_usage * days,
            "by_feed": feed_forecasts,
            "feeds_needing_reorder": [
                f for f, d in feed_forecasts.items()
                if d["will_run_out"]
            ],
        }

    def usage_by_destination(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze usage by destination.

        Args:
            start_date: Analysis start
            end_date: Analysis end

        Returns:
            Dict of destination -> {feed: quantity}
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        usages = [
            u for u in self.inventory._usage
            if start_date <= u.date <= end_date
        ]

        by_dest: Dict[str, Dict[str, float]] = {}

        for usage in usages:
            dest = usage.destination or "unspecified"
            if dest not in by_dest:
                by_dest[dest] = {}

            if usage.feed_name not in by_dest[dest]:
                by_dest[dest][usage.feed_name] = 0

            by_dest[dest][usage.feed_name] += usage.quantity

        return by_dest

    def price_trend(
        self,
        feed_name: str,
        months: int = 6,
    ) -> Dict[str, Any]:
        """
        Analyze price trend for feed.

        Args:
            feed_name: Feed type
            months: Months to analyze

        Returns:
            Dict with price trend analysis
        """
        cutoff = date.today() - timedelta(days=months * 30)

        purchases = [
            p for p in self.inventory._purchases
            if p.feed_name == feed_name and p.date >= cutoff
        ]

        if not purchases:
            return {"status": "no_data"}

        # Group by month
        monthly_prices: Dict[str, List[float]] = {}
        for purchase in purchases:
            month_key = purchase.date.strftime("%Y-%m")
            if month_key not in monthly_prices:
                monthly_prices[month_key] = []
            monthly_prices[month_key].append(purchase.unit_price)

        # Calculate monthly averages
        monthly_avg = {
            m: np.mean(prices)
            for m, prices in monthly_prices.items()
        }

        # Calculate trend
        if len(monthly_avg) >= 2:
            months_sorted = sorted(monthly_avg.keys())
            prices = [monthly_avg[m] for m in months_sorted]
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]

            if slope > 0.01:
                trend = "increasing"
            elif slope < -0.01:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            slope = 0
            trend = "insufficient_data"

        return {
            "feed_name": feed_name,
            "period_months": months,
            "n_purchases": len(purchases),
            "monthly_averages": monthly_avg,
            "overall_avg": np.mean([p.unit_price for p in purchases]),
            "min_price": min(p.unit_price for p in purchases),
            "max_price": max(p.unit_price for p in purchases),
            "trend": trend,
            "slope": slope,
        }
